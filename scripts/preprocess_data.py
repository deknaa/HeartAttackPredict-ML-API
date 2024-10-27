import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    #load dataset
    df = pd.read_csv(file_path)

    # preprocess data (delete rows with missing values)
    df = df.dropna()

    # memisahkan features dan target
    X = df.drop(columns=['target'])
    y = df['target']

    return train_test_split(X, y, test_size=0.2, random_state=42)