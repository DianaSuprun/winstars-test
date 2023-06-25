from sklearn.model_selection import train_test_split
import pandas as pd
from constants import CSV_PATH, RANDOM_STATE

df = pd.read_csv(CSV_PATH)
ship_df = df.copy()
ship_df['NumberOfShips'] = ship_df['EncodedPixels'].notnull().astype(int)
ship_df['EncodedPixels'] = ship_df['EncodedPixels'].replace(0, '')
ship_df = ship_df.groupby('ImageId').sum().reset_index()
ship_df["EncodedPixels"] = ship_df["EncodedPixels"].apply(lambda x: x if x != 0 else "")
df = df.fillna("")


def undersample_zeros(df):
    zeros = df[df['NumberOfShips'] == 0].sample(n=25_000, random_state = RANDOM_STATE, replace=True)
    nonzeros = df[df['NumberOfShips'] != 0]
    return pd.concat((nonzeros, zeros))


train_ships, valid_ships = train_test_split(ship_df,
                 test_size = 0.3,
                 stratify = ship_df['NumberOfShips'])
train_ships = undersample_zeros(train_ships)
valid_ships = undersample_zeros(valid_ships)


