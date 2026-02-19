import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/workspaces/Human-vs-ML-Project/data/hawks_new.csv')

def plot_comparison(x_axis, y_axis):
    plt.scatter(x_axis, y_axis)

    plt.savefig('')
    plt.close()

# plot_comparison('Wing','Weight')