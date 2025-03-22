import pandas

file = "landmarks_output.csv"

df = pandas.read_csv(file)

print(df.head())
# Check columns for the first frame's right-hand data
print("Right Hand Columns:", [col for col in df.columns if "right_hand" in col])

# Check first frame's right-hand coordinates
sample_row = df.iloc[0]
print("Right Hand X (Frame 0):", [sample_row[f"right_hand_{i}_x"] for i in range(21)])
print("Right Hand Y (Frame 0):", [sample_row[f"right_hand_{i}_y"] for i in range(21)])