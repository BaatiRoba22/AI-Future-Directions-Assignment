import matplotlib.pyplot as plt
import tensorflow as tf

# Load history (you must save this from train_model.py if running separately)
# If run in the same session, use the history object directly

# Example (you can move this to train_model.py if needed):
# Save accuracy history
# import pickle
# with open('history.pkl', 'wb') as f:
#     pickle.dump(history.history, f)

# For now, just plot inside train_model.py after training ends
# Below is the plotting code:

def plot_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy Over Epochs')
    plt.tight_layout()
    plt.savefig('accuracy_plot.png')
    plt.show()

# Use this if calling from same file
# plot_accuracy(history)
