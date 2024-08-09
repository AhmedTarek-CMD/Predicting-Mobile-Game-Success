import matplotlib.pyplot as plt

class GraphPlotter:
    def __init__(self, accuracies, train_times, test_times, hyperparameters):
        self.accuracies = accuracies
        self.train_times = train_times
        self.test_times = test_times
        self.hyperparameters = hyperparameters

    def plot_accuracy(self):
        plt.figure(figsize=(10, 10))  # Set the figure size to 8x6 inches
        plt.bar(self.hyperparameters, self.accuracies)
        plt.title('Classification Accuracy')
        plt.xlabel('Hyperparameters')
        plt.ylabel('Accuracy')
        plt.show()

    def plot_train_time(self):
        plt.figure(figsize=(10, 10))  # Set the figure size to 8x6 inches
        plt.bar(self.hyperparameters, self.train_times)
        plt.title('Total Training Time')
        plt.xlabel('Hyperparameters')
        plt.ylabel('Time (seconds)')
        plt.show()

    def plot_test_time(self):
        plt.figure(figsize=(10, 10))  # Set the figure size to 8x6 inches
        plt.bar(self.hyperparameters, self.test_times)
        plt.title('Total Test Time')
        plt.xlabel('Hyperparameters')
        plt.ylabel('Time (seconds)')
        plt.show()
