import matplotlib.pyplot as plt
import pandas as pd

from human_algorithm.human_classifier import human_classify
from sklearn.model_selection import train_test_split

df = pd.read_csv('/workspaces/Human-vs-ML-Project/data/hawks_new.csv')
training_df,test_df = train_test_split(df, test_size=0.3, random_state=32)

test_df['human_prediction'] = test_df['Hallux'].apply(human_classify)
test_df['correct'] = test_df['human_prediction'] == test_df['Species']
accuracy = test_df['correct'].mean()

conf_matrix = pd.crosstab(
    test_df['Species'],
    test_df['human_prediction'],
    rownames=['Actual'],
    colnames=['Predicted']
)
print(conf_matrix)
print(f'Accuracy of the human model: {accuracy:.2%}')

wrong_results = test_df[test_df['correct'] == False]
correct_results = test_df[test_df['correct'] == True]

plt.scatter(
    correct_results['Hallux'],
    correct_results['Weight'],
    c='green',
    marker='o',
    label='Correct',
)
plt.scatter(
    wrong_results['Hallux'],
    wrong_results['Weight'],
    c='red',
    marker='x',
    label='Wrong',
)

plt.xlabel('Hallux')
plt.ylabel('Weight')
plt.title('Human Algorithm Prediction Results')
plt.legend(title='Prediction Result')
plt.savefig('human_algorithm/plots/human_algorithm_results.png')