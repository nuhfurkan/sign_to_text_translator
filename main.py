import pandas

file = "landmarks_output.csv"

df = pandas.read_csv(file)

print(df.shape)