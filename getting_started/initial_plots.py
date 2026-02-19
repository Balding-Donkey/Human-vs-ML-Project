import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/workspaces/Human-vs-ML-Project/data/hawks_new.csv')
sampled_df = df.sample(frac=0.7, random_state=32)

coopers = sampled_df[sampled_df['Species'] == 'CH']
redtailed = sampled_df[sampled_df['Species'] == 'RT']
sharpshinned = sampled_df[sampled_df['Species'] == 'SS']


def plot_comparison(x_axis, y_axis):
    plt.scatter(coopers[x_axis], coopers[y_axis], label='Coopers')
    plt.scatter(redtailed[x_axis], redtailed[y_axis], label='Red Tailed')
    plt.scatter(sharpshinned[x_axis], sharpshinned[y_axis], label='Sharp Shinned')

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{x_axis} compared to {y_axis}')
    plt.legend()


    plt.savefig(f'/workspaces/Human-vs-ML-Project/getting_started/plots/{x_axis}_{y_axis}')
    plt.close()

# All 6 possible comparisons between 4 measurements
plot_comparison('Wing','Weight')
plot_comparison('Tail','Hallux')
plot_comparison('Hallux','Weight')
plot_comparison('Wing','Hallux')
plot_comparison('Tail','Weight')
plot_comparison('Wing','Tail')