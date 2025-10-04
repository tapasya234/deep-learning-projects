import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def plot_results(
    metrics,
    title=None,
    yLabel=None,
    yLim=None,
    xLim=None,
    metricName=None,
    colour=None,
):
    fig, axis = plt.subplots(figsize=(15, 8))

    for index, metric in enumerate(metrics):
        axis.plot(metric, color=colour[index])

    plt.xlabel("Epochs")
    plt.xlim(xLim)

    plt.ylabel(yLabel)
    plt.ylim(yLim)

    plt.title(title)

    # Tailor x-axis tick marks
    axis.xaxis.set_major_locator(MultipleLocator(5))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    axis.xaxis.set_minor_locator(MultipleLocator(1.1))

    plt.grid(True)
    plt.legend(metricName)
    plt.show()
    plt.close()
