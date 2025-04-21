# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import argparse
# from matplotlib.widgets import Slider


# def show_data(motifs_data, data_path, m, selected_columns:list[int] = None):
#     data = pd.read_csv(motifs_data)
#     motifs_idx= data["motifs_idx"]
#     nn_idx = data["nn_idx"]
    
#     df = pd.read_csv(data_path)
    
#     if selected_columns is None:
#         selected_columns = enumerate(df.columns)
#     else :
#         selected_columns = [(i, df.columns[i]) for i in selected_columns]

#     data.drop(columns=["motifs_idx", "nn_idx"], inplace=True)
#     mps = data.to_numpy()
    
#     print(motifs_idx)
#     print(nn_idx)
#     '''---------------------------'''
    
#     window_size = 500
#     max_time = df.shape[0]  # assuming time axis is length of df
#     start = 0

#     fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 10))
#     plt.subplots_adjust(bottom=0.15)  # leave space for slider

#     for k, dim_name in selected_columns:
#         axs[k].set_ylabel(dim_name, fontsize='20')
#         axs[k].plot(df[dim_name])
#         axs[k].set_xlabel('Time', fontsize ='20')
    
#         axs[k + mps.shape[0]].set_ylabel(dim_name.replace('T', 'P'), fontsize='20')
#         axs[k + mps.shape[0]].plot(mps[k], c='orange')
#         axs[k + mps.shape[0]].set_xlabel('Time', fontsize ='20')
    
#         axs[k].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
#         axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
#         axs[k + mps.shape[0]].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
#         axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')
        
#         if dim_name != 'T3':
#             axs[k].plot(range(motifs_idx[k], motifs_idx[k] + m), df[dim_name].iloc[motifs_idx[k] : motifs_idx[k] + m], c='red', linewidth=4)
#             axs[k].plot(range(nn_idx[k], nn_idx[k] + m), df[dim_name].iloc[nn_idx[k] : nn_idx[k] + m], c='red', linewidth=4)
#             axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='red')
#             axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')
#         else:
#             axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='black')
#             axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='black')
    
#     # Add horizontal slider
#     slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
#     scroll_slider = Slider(slider_ax, 'Scroll Time', 0, max_time - window_size, valinit=0, valstep=1)
    
#     # Update all subplots when slider is moved
#     def update(val):
#         start = int(scroll_slider.val)
#         end = start + window_size
#         for ax in axs:
#             ax.set_xlim(start, end)
#         fig.canvas.draw_idle()
    
#     scroll_slider.on_changed(update)
    
#     # Set initial x-limits
#     for ax in axs:
#         ax.set_xlim(start, start + window_size)

#     plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import argparse

def show_data(motifs_data_path, data_path, m, selected_columns: list[int] = None):
    # Load data
    motif_data = pd.read_csv(motifs_data_path)
    motifs_idx = motif_data["motifs_idx"].to_numpy()
    nn_idx = motif_data["nn_idx"].to_numpy()

    motif_data.drop(columns=["motifs_idx", "nn_idx"], inplace=True)
    mps = motif_data.to_numpy()

    df = pd.read_csv(data_path)

    # Column selection
    if selected_columns is None:
        selected_columns = list(range(min(5, df.shape[1])))  # plot only 5 by default
    column_names = [df.columns[i] for i in selected_columns]

    window_size = 500
    max_time = df.shape[0]
    start = 0

    num_dims = len(selected_columns)
    fig, axs = plt.subplots(num_dims * 2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 6))
    plt.subplots_adjust(bottom=0.15)

    signal_lines = []
    mps_lines = []

    for plot_idx, (k, dim_name) in enumerate(zip(selected_columns, column_names)):
        signal_ax = axs[plot_idx]
        mps_ax = axs[plot_idx + num_dims]

        # Plot full data, initially visible part is limited via xlim
        signal_line, = signal_ax.plot(df[dim_name], label=dim_name)
        mps_line, = mps_ax.plot(mps[k], color='orange', label=f"{dim_name} MP")

        signal_ax.set_ylabel(dim_name, fontsize=12)
        mps_ax.set_ylabel(dim_name.replace("T", "P"), fontsize=12)

        # Optional motif/nearest neighbor markers
        signal_ax.axvline(x=motifs_idx[1], linestyle="dashed", color='black')
        signal_ax.axvline(x=nn_idx[1], linestyle="dashed", color='black')
        mps_ax.axvline(x=motifs_idx[1], linestyle="dashed", color='black')
        mps_ax.axvline(x=nn_idx[1], linestyle="dashed", color='black')

        if dim_name != 'T3':
            signal_ax.plot(
                range(motifs_idx[k], motifs_idx[k] + m),
                df[dim_name].iloc[motifs_idx[k]:motifs_idx[k] + m],
                color='red', linewidth=4
            )
            signal_ax.plot(
                range(nn_idx[k], nn_idx[k] + m),
                df[dim_name].iloc[nn_idx[k]:nn_idx[k] + m],
                color='red', linewidth=4
            )
            mps_ax.plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", color='red', markersize=10)
            mps_ax.plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", color='red', markersize=10)
        else:
            mps_ax.plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", color='black', markersize=10)
            mps_ax.plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", color='black', markersize=10)

        signal_lines.append(signal_line)
        mps_lines.append(mps_line)

    # Slider setup
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    scroll_slider = Slider(slider_ax, 'Scroll Time', 0, max_time - window_size, valinit=0, valstep=1)

    def update(val):
        start = int(scroll_slider.val)
        end = start + window_size
        for ax in axs:
            ax.set_xlim(start, end)
        fig.canvas.draw_idle()

    scroll_slider.on_changed(update)

    # Initial view window
    for ax in axs:
        ax.set_xlim(start, start + window_size)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize motifs")

    parser.add_argument("--motifs_data", type=str, required=True, help="Path to the motifs data")
    parser.add_argument("--data", type=str, required=True, help="Path to the data")
    parser.add_argument("--m", type=int, default=30, help="Number of motifs to visualize")
    parser.add_argument("--selected_columns", type=int, nargs='+', help="Indices of columns to visualize")

    args = parser.parse_args()    
    show_data(args.motifs_data, args.data, args.m, selected_columns=args.selected_columns)