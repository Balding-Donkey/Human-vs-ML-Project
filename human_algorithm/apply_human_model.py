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

# plt.scatter(
#     test_df['Hallux'],
#     test_df['Weight'],
#     'correct'
# )