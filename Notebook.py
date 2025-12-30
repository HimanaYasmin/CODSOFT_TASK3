import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("IRIS.csv")
df.head()
df.shape
df.columns
df.info()
df.isnull().sum()
df['species'].value_counts()
sns.pairplot(df, hue="species")
plt.show()
df.hist(figsize=(10,8))
plt.show()
sns.boxplot(data=df)
plt.show()
X = df.drop("species", axis=1)
y = df["species"]
X.head()
y.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train.shape
X_test.shape
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred[:5]
y_test[:5]
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# New flower measurements
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)
predicted_species = le.inverse_transform(prediction)

print("Predicted Species:", predicted_species[0])

import joblib

joblib.dump(model, "iris_model.pkl")
