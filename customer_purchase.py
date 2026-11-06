
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("customer_purchase.csv")

print("Loaded dataset shape:", df.shape)
df.info()

print("\nMissing values per column:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())


X = df.drop(columns=["Purchased"])
y = df["Purchased"]

print("Features:", X.columns.tolist())
print("Target:", y.name)
print("X shape:", X.shape)
print("y shape:", y.shape)

class_counts = y.value_counts().sort_index()

class_summary = pd.DataFrame({
    "Customers": class_counts,
    "Percentage": (class_counts / len(y) * 100).round(1)
})

print(class_summary)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Total records:", len(y))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])

print(pd.DataFrame({
    "Total": y.value_counts(),
    "Train": y_train.value_counts(),
    "Test": y_test.value_counts()
}).sort_index())