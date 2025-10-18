import matplotlib.pyplot as plt


def plotLoss(epoch_train_loss, epoch_test_loss):
    # Plot loss
    plt.rcParams["figure.figsize"] = (10, 6)
    x = range(len(epoch_train_loss))

    plt.figure
    plt.plot(x, epoch_train_loss, color="r", label="train loss")
    plt.plot(x, epoch_test_loss, color="b", label="validation loss")
    plt.xlabel("epoch no.")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()


def plotAccuracy(epoch_train_loss, epoch_train_acc, epoch_test_acc):
    # Plot Accuracy
    plt.rcParams["figure.figsize"] = (10, 6)
    x = range(len(epoch_train_loss))

    plt.figure
    plt.plot(x, epoch_train_acc, color="r", label="train accuracy")
    plt.plot(x, epoch_test_acc, color="b", label="validation accuracy")
    plt.xlabel("epoch no.")
    plt.ylabel("accuracy")
    plt.legend(loc="center right")
    plt.title("Training and Validation Accuracy")
    plt.show()
