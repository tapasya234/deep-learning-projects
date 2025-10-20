import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def plotResults(
    metrics,
    xLabel="Epoch",
    xLim=None,
    yLabel=None,
    yLim=None,
    metricName=None,
    colour=None,
):
    fig, ax = plt.subplots(figsize=(18, 5))

    if not (isinstance(metricName, list) or isinstance(metricName, tuple)):
        metrics = [
            metrics,
        ]
        metricName = [
            metricName,
        ]

    for index, metric in enumerate(metrics):
        ax.plot(metric, color=colour[index])

    plt.xlabel(xLabel)
    plt.xlim(xLim)

    plt.ylabel(yLabel)
    plt.ylim(yLim)

    plt.title(yLabel)

    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.grid(True)
    plt.legend(metricName)
    plt.show()
    plt.close()
