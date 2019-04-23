import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_to_log = sys.argv[1] if len(sys.argv) is 2 else 'X:\Project\logs\defstage-2.log'

history = pd.read_csv(path_to_log, index_col=0)

plt.plot(history['loss'], label='training')
plt.plot(history['val_loss'], label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss history')
plt.savefig('logs/loss.png')
plt.clf()

plt.plot(history['binary_accuracy'], label='training')
plt.plot(history['val_binary_accuracy'], label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('acc history')
plt.savefig('logs/acc.png')
plt.clf()

print('Best validation loss: {:.5f}'.format(np.amin(history['val_loss'])))
print('Best validation accuracy: {:.5f}'.format(np.amax(history['val_binary_accuracy'])))