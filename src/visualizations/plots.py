import matplotlib.pyplot as plt

def plot_record(input, output, figsize=(9, 4),show_axis=False, cmap=None):
    if show_axis:
        axis = 'on'
    else:
        axis = 'off'

    plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.imshow(input[0], cmap=cmap)
    plt.axis(axis)
    plt.title('Input 1')

    plt.subplot(1, 3, 2)
    plt.imshow(input[1], cmap=cmap)
    plt.axis(axis)
    plt.title('Input 2')

    plt.subplot(1, 3, 3)
    plt.imshow(output, cmap=cmap)
    plt.axis(axis)
    plt.title('Output')

    plt.tight_layout()
    plt.show()

def plot_prediction(y_true, y_pred, cmap=None, show_axis=False, text=None):
    if show_axis:
        axis = 'on'
    else:
        axis = 'off'

    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(y_true, cmap=cmap)
    plt.axis(axis)
    plt.title('True')

    plt.subplot(1, 2, 2)
    plt.imshow(y_pred, cmap=cmap)
    plt.axis(axis)
    if text is None:
      plt.title('Prediction')
    else:
      plt.title(f'Prediction: ({text})')

    plt.tight_layout()
    plt.show()