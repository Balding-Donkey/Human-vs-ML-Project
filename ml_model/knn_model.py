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

# y_pred['correct'] = y_pred == y_test
# pred_wrong = y_pred[y_pred['correct'] == False]
# pred_correct = y_pred[y_pred['correct'] == True]

# y_test['correct'] = y_pred == y_test
# pred_wrong = y_pred[y_pred['correct'] == False]
# pred_correct = y_pred[y_pred['correct'] == True]

# plt.scatter(
#     correct_results['Hallux'],
#     correct_results['Weight'],
#     c='green',
#     marker='o',
#     label='Correct',
# )
# plt.scatter(
#     wrong_results['Hallux'],
#     wrong_results['Weight'],
#     c='red',
#     marker='x',
#     label='Wrong',
# )

# plt.xlabel('Hallux')
# plt.ylabel('Weight')
# plt.title('Human Algorithm Prediction Results')
# plt.legend(title='Prediction Result')
# plt.savefig('human_algorithm/plots/human_algorithm_results.png')