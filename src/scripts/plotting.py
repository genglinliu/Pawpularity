# plotting loss curve

def make_plots(step_hist, loss_hist):
    plt.plot(step_hist, loss_hist)
    plt.xlabel('train_iterations')
    plt.ylabel('Loss')
    plt.title(experiment_name)
    plt.show()
    plt.savefig(experiment_name)
    plt.clf()