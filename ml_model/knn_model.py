import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv('/workspaces/Human-vs-ML-Project/data/hawks_new.csv')

x = df[['Hallux','Weight']]
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=32,
    stratify=y
)

k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
y_train_pred = knn.predict(x_train)


accuracy = (y_pred == y_test).mean()
conf_matrix = pd.crosstab(
    y_test,
    y_pred,
    rownames=['Actual'],
    colnames=['Predicted']
)

print(conf_matrix)
print(f'Accuracy of the KNN model (k={k}): {accuracy:.2%}')


train_df = x_train.copy()
train_df['species'] = y_train
train_df['pred'] = y_train_pred
train_df['correct'] = train_df['pred'] == train_df['species']
train_pred_wrong = train_df[train_df['correct'] == False]
train_pred_correct = train_df[train_df['correct'] == True]

# Plot the training data
plt.scatter(
    train_pred_correct['Hallux'],
    train_pred_correct['Weight'],
    c='green',
    marker='o',
    label='Correct',
)
plt.scatter(
    train_pred_wrong['Hallux'],
    train_pred_wrong['Weight'],
    c='red',
    marker='x',
    label='Wrong',
)

plt.xlabel('Hallux')
plt.ylabel('Weight')
plt.title('KNN Algorithm Training Results')
plt.legend(title='Prediction')
plt.savefig('ml_model/plots/knn_train_results.png')
plt.close()



test_df = x_test.copy()
test_df['species'] = y_test
test_df['pred'] = y_pred
test_df['correct'] = test_df['pred'] == test_df['species']
pred_wrong = test_df[test_df['correct'] == False]
pred_correct = test_df[test_df['correct'] == True]

# Plot the test data
plt.scatter(
    pred_correct['Hallux'],
    pred_correct['Weight'],
    c='green',
    marker='o',
    label='Correct',
)
plt.scatter(
    pred_wrong['Hallux'],
    pred_wrong['Weight'],
    c='red',
    marker='x',
    label='Wrong',
)

plt.xlabel('Hallux')
plt.ylabel('Weight')
plt.title('KNN Algorithm Test Results')
plt.legend(title='Prediction')
plt.savefig('ml_model/plots/knn_test_results.png')
plt.close()