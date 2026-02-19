import matplotlib.pyplot as plt
import pandas as pd

from getting_started.human_classifier import human_classify

df = pd.read_csv('/workspaces/Human-vs-ML-Project/data/hawks_new.csv')
sampled_df = df.sample(frac=0.7, random_state=32)