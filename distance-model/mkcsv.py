import pandas as pd

df = pd.DataFrame({
    'y_coordinate': [373, 333, 309, 292, 281, 274, 267, 265, 263, 262, 261, 257],
    'distance': [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 35]
})
df.to_csv("a.csv")
