import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

df = pd.read_csv('twitter_training.csv')
df.columns = ['ID', 'Subject', 'Sentiment', 'Tweet']
df = df.dropna(subset=['Tweet'])

X = df['Tweet']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=200, random_state=42))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

custom_tweets = [
    "This game is the best I’ve ever played!",   
    "This game is terrible and a waste of time.", 
    "I have no strong feelings about this game.", 
    "I just bought a new phone."                 
]

custom_predictions = model.predict(custom_tweets)

print("\nCustom Tweet Predictions:")
for tweet, prediction in zip(custom_tweets, custom_predictions):
    print(f"Tweet: '{tweet}' => Sentiment: {prediction}")