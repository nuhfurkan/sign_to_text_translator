import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformer import Transformer
from bpe import BasicTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import random
import torchmetrics.text

# Constants
START_TOKEN = '<STR>'
PADDING_TOKEN = '<PAD>'
END_TOKEN = '<EOS>'
NEG_INFTY = -1e9
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TextDataset(Dataset):
    def __init__(self, source_sentences, target_sentences):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return self.source_sentences[idx], self.target_sentences[idx]


def create_masks(src_batch, tgt_batch, max_sequence_length):
    num_sentences = len(src_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        src_sentence_length, tgt_sentence_length = len(src_batch[idx]), len(tgt_batch[idx])
        src_chars_to_padding_mask = np.arange(src_sentence_length + 1, max_sequence_length)
        tgt_chars_to_padding_mask = np.arange(tgt_sentence_length + 1, max_sequence_length)

        encoder_padding_mask[idx, :, src_chars_to_padding_mask] = True
        encoder_padding_mask[idx, src_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, tgt_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, tgt_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, src_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, tgt_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)

    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


def decode_predictions(predictions, tokenizer, end_token_idx):
    """
    Decode model predictions into readable text using the tokenizer's decoder
    """
    # Get most likely token indices (greedy decoding)
    token_indices = torch.argmax(predictions, dim=1).cpu().numpy()

    # Find where the sequence ends (at the end token)
    end_pos = len(token_indices)
    for i, idx in enumerate(token_indices):
        if idx == end_token_idx:
            end_pos = i
            break

    # Use the tokenizer's decode method with the sequence up to end token
    decoded_text = tokenizer.decode(token_indices[:end_pos])

    return decoded_text


def translate(model, src_sentence, tgt_tokenizer, max_sequence_length):
    """
    Translate a source sentence using the trained model
    """
    model.eval()
    src_sentence = (src_sentence,)
    tgt_sentence = ("",)

    with torch.no_grad():
        for i in range(max_sequence_length):
            # Create masks
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                src_sentence, tgt_sentence, max_sequence_length)

            # Generate prediction
            predictions = model(
                src_sentence,
                tgt_sentence,
                encoder_self_attention_mask.to(DEVICE),
                decoder_self_attention_mask.to(DEVICE),
                decoder_cross_attention_mask.to(DEVICE),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=False
            )

            # Get next token
            next_token_prob_distribution = predictions[0][i]
            next_token_index = torch.argmax(next_token_prob_distribution).item()

            # Convert token index to actual token
            next_token = tgt_tokenizer.decode([next_token_index])

            # Add next token to target sentence
            tgt_sentence = (tgt_sentence[0] + next_token,)

            # Stop if end token is generated or reaching max length
            if next_token_index == tgt_tokenizer.special_tokens[END_TOKEN] or i == max_sequence_length - 1:
                break

    # Clean up the output by removing special tokens
    result = tgt_sentence[0]
    if END_TOKEN in result:
        result = result[:result.index(END_TOKEN)]

    return result


def train_model(model, train_loader, val_loader, optimizer, scheduler, scheduler_type,
                criterion, target_padding_idx, epochs=5, clip_grad_norm=1.0):
    model.train()
    model.to(DEVICE)

    best_val_loss = float('inf')
    no_improvement_count = 0
    patience = 5  # Early stopping patience

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0

        # Training loop
        model.train()
        for batch_num, batch in enumerate(train_loader):
            src_batch, tgt_batch = batch

            # Create masks
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                src_batch, tgt_batch, model.decoder.sentence_embedding.max_sequence_length)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(
                src_batch,
                tgt_batch,
                encoder_self_attention_mask.to(DEVICE),
                decoder_self_attention_mask.to(DEVICE),
                decoder_cross_attention_mask.to(DEVICE),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=True
            )

            # Get target labels
            labels = model.decoder.sentence_embedding.batch_tokenize(tgt_batch, start_token=False, end_token=True)

            # Calculate loss
            loss = criterion(
                predictions.view(-1, predictions.size(-1)).to(DEVICE),
                labels.view(-1).to(DEVICE)
            )

            # Mask out padding tokens in loss calculation
            valid_indices = labels.view(-1) != target_padding_idx
            loss = loss[valid_indices].mean()

            # Backward pass
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            total_loss += loss.item()

            # Print progress
            if batch_num % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_num}, Loss: {loss.item():.4f}")

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)

        # Validation step
        val_loss, bleu_score, chrf_score, wer_score = evaluate(model, val_loader, criterion, target_padding_idx)

        # Update learning rate based on validation loss
        if scheduler_type == "plateau":
            scheduler.step(val_loss)
        elif scheduler_type in ["cosine", "linear"]:
            scheduler.step()

        # Print epoch stats
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"BLEU Score: {bleu_score:.4f}, CHRF Score: {chrf_score:.4f}, WER Score: {wer_score:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'transformer_best_model.pt')
            print(f"Model saved (best validation loss: {best_val_loss:.4f})")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= patience:
            print(f"Early stopping after {epoch + 1} epochs without improvement")
            break

        # Save checkpoint for every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, f'transformer_checkpoint_epoch_{epoch + 1}.pt')

    return model


def evaluate(model, val_loader, criterion, target_padding_idx):
    """
    Evaluate the model on the validation dataset
    """
    model.eval()
    total_loss = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            src_batch, tgt_batch = batch

            # Create masks
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                src_batch, tgt_batch, model.decoder.sentence_embedding.max_sequence_length)

            # Forward pass
            predictions = model(
                src_batch,
                tgt_batch,
                encoder_self_attention_mask.to(DEVICE),
                decoder_self_attention_mask.to(DEVICE),
                decoder_cross_attention_mask.to(DEVICE),
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=True
            )

            # Get target labels
            labels = model.decoder.sentence_embedding.batch_tokenize(tgt_batch, start_token=False, end_token=True)

            # Calculate loss
            loss = criterion(
                predictions.view(-1, predictions.size(-1)).to(DEVICE),
                labels.view(-1).to(DEVICE)
            )

            # Mask out padding tokens in loss calculation
            valid_indices = labels.view(-1) != target_padding_idx
            loss = loss[valid_indices].mean()

            total_loss += loss.item()

            # Get predicted tokens for metrics calculation
            pred_indices = torch.argmax(predictions, dim=-1)

            # Convert token indices to text for metrics calculation
            for i in range(len(src_batch)):
                # Get prediction and target sequences
                pred_tokens = pred_indices[i].cpu().tolist()
                target_tokens = labels[i].cpu().tolist()

                # Filter out padding tokens
                pred_tokens = [t for t in pred_tokens if t != target_padding_idx]
                target_tokens = [t for t in target_tokens if t != target_padding_idx]

                # Decode using tokenizer
                pred_text = model.decoder.sentence_embedding.tokenizer.decode(pred_tokens)
                target_text = model.decoder.sentence_embedding.tokenizer.decode(target_tokens)

                # Add to predictions and targets lists
                all_predictions.append(pred_text)
                all_targets.append(target_text)  # Just the string, not a list containing the string

    avg_loss = total_loss / len(val_loader)

    # Create metrics here to ensure they're fresh for each evaluation
    bleu_metric = torchmetrics.text.BLEUScore().to(DEVICE)
    chrf_metric = torchmetrics.text.CHRFScore().to(DEVICE)
    wer_metric = torchmetrics.text.WordErrorRate().to(DEVICE)

    # Wrap targets in list-of-lists for BLEU and CHRF
    wrapped_targets = [[t] for t in all_targets]

    # BLEU
    bleu_metric.update(all_predictions, wrapped_targets)
    bleu_score = bleu_metric.compute()

    # CHRF
    chrf_metric.update(all_predictions, wrapped_targets)
    chrf_score = chrf_metric.compute()

    # WER (expects flat list of strings)
    wer_metric.update(all_predictions, all_targets)
    wer_score = wer_metric.compute()

    model.train()
    return avg_loss, bleu_score, chrf_score, wer_score


def main():
    parser = argparse.ArgumentParser(description="Train a transformer model for translation")
    parser.add_argument("--src_file", type=str, required=True, help="Path to source language file")
    parser.add_argument("--tgt_file", type=str, required=True, help="Path to target language file")
    parser.add_argument("--val_src_file", type=str, help="Path to validation source language file")
    parser.add_argument("--val_tgt_file", type=str, help="Path to validation target language file")
    parser.add_argument("--src_tokenizer", type=str, help="Path to source tokenizer file")
    parser.add_argument("--tgt_tokenizer", type=str, help="Path to target tokenizer file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--max_seq_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine", "linear", "none"],
                        help="Learning rate scheduler type")

    args = parser.parse_args()

    # Load train data
    with open(args.src_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip().lower() for line in f]

    with open(args.tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = [line.strip().lower() for line in f]

    # Load validation data if provided
    val_src_sentences = []
    val_tgt_sentences = []
    if args.val_src_file and args.val_tgt_file:
        with open(args.val_src_file, 'r', encoding='utf-8') as f:
            val_src_sentences = [line.strip().lower() for line in f]

        with open(args.val_tgt_file, 'r', encoding='utf-8') as f:
            val_tgt_sentences = [line.strip().lower() for line in f]

        print(f"Loaded {len(val_src_sentences)} validation examples")
    else:
        print("No validation files provided. Will split training data for validation.")

    # Limit data if needed
    max_samples = 100000
    src_sentences = src_sentences[:max_samples]
    tgt_sentences = tgt_sentences[:max_samples]

    # Create tokenizers
    src_tokenizer = BasicTokenizer()
    tgt_tokenizer = BasicTokenizer()

    if args.src_tokenizer:
        src_tokenizer.load(args.src_tokenizer)
    else:
        # Register special tokens
        special_tokens = {
            START_TOKEN: len(src_tokenizer.vocab),
            END_TOKEN: len(src_tokenizer.vocab) + 1,
            PADDING_TOKEN: len(src_tokenizer.vocab) + 2
        }
        src_tokenizer.register_special_tokens(special_tokens)
        # Train tokenizers on respective corpus
        print("Training source tokenizer...")
        src_tokenizer.train(''.join(src_sentences), args.vocab_size, verbose=True)
        src_tokenizer.save("src_tokenizer")


    if args.tgt_tokenizer:
        tgt_tokenizer.load(args.tgt_tokenizer)
    else:
        # Register special tokens
        special_tokens = {
            START_TOKEN: len(tgt_tokenizer.vocab),
            END_TOKEN: len(tgt_tokenizer.vocab) + 1,
            PADDING_TOKEN: len(tgt_tokenizer.vocab) + 2
        }
        tgt_tokenizer.register_special_tokens(special_tokens)
        # Train tokenizers on respective corpus
        print("Training target tokenizer...")
        tgt_tokenizer.train(''.join(tgt_sentences), args.vocab_size, verbose=True)
        tgt_tokenizer.save("tgt_tokenizer")

    # If no validation files were provided, split training data
    if not val_src_sentences:
        # Split data into training and validation sets
        data_size = len(src_sentences)
        val_size = int(data_size * 0.1)  # 10% for validation
        train_size = data_size - val_size

        # Shuffle and split the data
        indices = list(range(data_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_src = [src_sentences[i] for i in train_indices]
        train_tgt = [tgt_sentences[i] for i in train_indices]
        val_src = [src_sentences[i] for i in val_indices]
        val_tgt = [tgt_sentences[i] for i in val_indices]
    else:
        train_src = src_sentences
        train_tgt = tgt_sentences
        val_src = val_src_sentences
        val_tgt = val_tgt_sentences

    # Create dataset and dataloaders
    train_dataset = TextDataset(train_src, train_tgt)
    val_dataset = TextDataset(val_src, val_tgt)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Initialize model
    model = Transformer(
        d_model=args.d_model,
        ffn_hidden=args.d_model * 4,
        num_heads=args.num_heads,
        drop_prob=0.1,
        num_layers=args.num_layers,
        max_sequence_length=args.max_seq_len,
        kn_vocab_size=len(tgt_tokenizer.vocab) + len(tgt_tokenizer.special_tokens),
        sign_tokenizer=src_tokenizer,
        english_tokenizer=tgt_tokenizer,
        START_TOKEN=START_TOKEN,
        END_TOKEN=END_TOKEN,
        PADDING_TOKEN=PADDING_TOKEN
    )

    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.special_tokens[PADDING_TOKEN], reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # Configure learning rate scheduler
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    elif args.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler == "linear":
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs)

    # Train model with the chosen scheduler
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_type=args.scheduler,
        criterion=criterion,
        target_padding_idx=tgt_tokenizer.special_tokens[PADDING_TOKEN],
        epochs=args.epochs,
        clip_grad_norm=args.clip_grad
    )

    # Test translation examples
    test_sentences = [
        "hello world",
        "how are you doing today?",
        "this is a test sentence",
        "machine learning is fascinating"
    ]

    print("\nTesting translations:")
    for sentence in test_sentences:
        translation = translate(
            model,
            sentence,
            tgt_tokenizer,
            args.max_seq_len
        )
        print(f"Source: {sentence}")
        print(f"Translation: {translation}")
        print("-" * 50)


if __name__ == "__main__":
    main()