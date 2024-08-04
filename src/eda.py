import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(data):
    sns.pairplot(data)
    plt.show()


def plot_correlations(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    plt.show()
