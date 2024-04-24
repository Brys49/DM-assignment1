import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

today = datetime.datetime.today()
date_str = today.strftime("%Y-%m-%d_%H-%M-%S")

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
ids = test['PassengerId']

# Prepare the data

train.drop(['Name', 'Sex','Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

test.fillna(test.mean(), inplace=True)
train.dropna(inplace=True)

y = train['Survived']
X = train.drop(['Survived'], axis=1)

# Training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_train)

print(f'TRAIN\nPreprocessed and scaled: {accuracy_score(y_train, y_pred)}')

y_pred = model.predict(X_test)
print(f'TEST\nPreprocessed and scaled: {accuracy_score(y_test, y_pred)}')

# Predictions

final_predictions = model.predict(test)
final_predictions = final_predictions.astype(int)
print(final_predictions)
submission = pd.DataFrame({'PassengerId':ids,'Survived':final_predictions})
print(submission.shape)
submission.to_csv(f'submissions/basic_submission_{date_str}.csv', index=False)