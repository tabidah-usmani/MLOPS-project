import joblib, pandas as pd, requests
from sklearn.model_selection import train_test_split

model = joblib.load('models/model.pkl')
df = pd.read_csv('dataset/cleaned.csv')
X, y = df['clean_text'], df['label']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

correct = 0
fake_sent = 0
real_sent = 0

for i in range(len(X_test)):
    if fake_sent >= 5 and real_sent >= 5:
        break
    actual = 'REAL' if y_test.iloc[i] == 1 else 'FAKE'
    if actual == 'FAKE' and fake_sent >= 5:
        continue
    if actual == 'REAL' and real_sent >= 5:
        continue
    text = X_test.iloc[i]
    if len(text) < 100:
        continue
    r = requests.post('http://localhost:5000/predict', json={'text': text}, timeout=5)
    d = r.json()
    predicted = d['label']
    ok = predicted == actual
    if ok: correct += 1
    if actual == 'FAKE': fake_sent += 1
    else: real_sent += 1
    mark = 'CORRECT' if ok else 'WRONG'
    print(mark + ' | Pred:' + predicted + ' Actual:' + actual + ' ' + str(d['latency_ms']) + 'ms')

print('Accuracy: ' + str(correct) + '/10')
