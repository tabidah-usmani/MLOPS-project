import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows")

    # WELFake columns: title, text, label (0=fake, 1=real)
    df = df.dropna(subset=['text', 'label'])
    df['clean_text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)
    df = df[df['clean_text'].str.strip() != '']  # remove empty strings after cleaning
    df = df.dropna(subset=['clean_text'])         # remove any remaining NaN
    df['label'] = df['label'].astype(int)

    print(f"After cleaning: {len(df)} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df[['clean_text', 'label']]

if __name__ == "__main__":
    df = load_and_clean("dataset/WELFake_Dataset.csv")
    df.to_csv("dataset/cleaned.csv", index=False)
    print("Saved to dataset/cleaned.csv")