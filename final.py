import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

today = datetime.datetime.today()
date_str = today.strftime("%Y-%m-%d_%H-%M-%S")

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
combined = pd.concat([train, test], ignore_index=True)

def preprocessing(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the combined train and test datasets for a machine learning model.

    This function performs the following data preprocessing steps:
    - One-hot encodes the 'Sex' column
    - Imputes missing 'Age' values with the median age for the corresponding 'Sex' and 'Pclass'
    - Imputes missing 'Fare' values with the median fare
    - Fills missing 'Cabin' values with 'Unknown' and encodes the first character of the cabin number
    - Groups the 'Cabin' values into broader categories
    - Encodes the 'Embarked' column as numeric values
    - Creates one-hot encoded columns for 'Cabin', 'Embarked', and 'Pclass'
    - Creates a new 'Family_Size' feature by summing 'SibSp' and 'Parch' plus 1
    - Drops the 'Name', 'Ticket', 'SibSp', and 'Parch' columns

    Args:
        dataset (pd.DataFrame): The input dataset to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    dataset = dataset.copy()
    
    dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0})
    
    dataset['Age'] = dataset.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median())) 

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

    dataset['Cabin'] = dataset['Cabin'].fillna('Unknown')
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x[0])
    dataset['Cabin'] = dataset['Cabin'].map({'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 7})
    dataset['Cabin'] = dataset['Cabin'].replace(['A', 'B', 'C', 'T'], 'ABC') # only one record with deck T and it's a first-class passenger so he fits in this group
    dataset['Cabin'] = dataset['Cabin'].replace(['D', 'E'], 'DE')
    dataset['Cabin'] = dataset['Cabin'].replace(['F', 'G'], 'FG')

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) 

    dataset = pd.get_dummies(dataset, columns=['Cabin'], prefix='Cabin')
    dataset = pd.get_dummies(dataset, columns=['Embarked'], prefix='Embarked')
    dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pclass")

    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset = dataset.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch'], axis=1)

    return dataset


combined_preprocessed = preprocessing(combined)
print(combined_preprocessed.shape)
print(combined_preprocessed.head())

# Split to train and test

train_preprocessed = combined_preprocessed[combined_preprocessed['Survived'].notnull()].drop('PassengerId', axis=1)
test_preprocessed = combined_preprocessed[combined_preprocessed['Survived'].isnull()].drop('Survived', axis=1)
ids = test_preprocessed.PassengerId
test_preprocessed = test_preprocessed.drop('PassengerId', axis=1)

y = train_preprocessed.Survived
X = train_preprocessed.drop(['Survived'], axis=1)

# Scaling

columns_to_scale = ['Age', 'Fare', 'Family_Size'] # only scale the non-categorical columns
columns_not_to_scale = X.columns.difference(columns_to_scale)
columns_not_to_scale_test = test_preprocessed.columns.difference(columns_to_scale)

# Scale the desired columns
scaler = StandardScaler()
X_scaled_columns = scaler.fit_transform(X[columns_to_scale])
X_scaled_columns = pd.DataFrame(X_scaled_columns, columns=columns_to_scale, index=X.index)
X_scaled = pd.concat([X_scaled_columns, X[columns_not_to_scale]], axis=1)

test_scaled_columns = scaler.fit_transform(test_preprocessed[columns_to_scale])
test_scaled_columns = pd.DataFrame(test_scaled_columns, columns=columns_to_scale, index=test_preprocessed.index)
test_final = pd.concat([test_scaled_columns, test_preprocessed[columns_not_to_scale_test]], axis=1)

# Training

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_train)

print(f'TRAIN\nPreprocessed and scaled: {accuracy_score(y_train, y_pred)}')

y_pred = model.predict(X_test)
print(f'TEST\nPreprocessed and scaled: {accuracy_score(y_test, y_pred)}')

# Predictions

final_predictions = model.predict(test_final)
final_predictions = final_predictions.astype(int)
print(final_predictions)
submission = pd.DataFrame({'PassengerId':ids,'Survived':final_predictions})
print(submission.shape)
submission.to_csv(f'submissions/submission_{date_str}.csv', index=False)
