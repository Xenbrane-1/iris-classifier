import os
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data       # shape (150, 4)
y = iris.target     # shape (150,)
print(iris.feature_names, iris.target_names)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# This will use 80% of the data for training and 20% for testing (120 samples and 30 test samples)
# Random state is set for reproducibility

# Initialise the model: We import the Decision Tree Classifier and create an instance of it. 
model = DecisionTreeClassifier(random_state=42)

# Train (fit) the model: We train the decision tree using the training data: )
model.fit(X_train, y_train)

# Make predictions: We use the trained model to make predictions on the test set.
y_pred = model.predict(X_test)
print("Predictions", y_pred[:5])
print("True labels", y_test[:5])

# Evaluating the model 

# Accuracy 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.xlabel("Predicted")
plt.ylabel("True")

# Save figure
plt.savefig("outputs/confusion_matrix.png")
plt.show()



