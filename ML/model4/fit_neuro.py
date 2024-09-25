import numpy as np
from neuron import SingleNeuron

X = np.array([  [-2, -1, 1],
                [25, 6, 0],
                [17, 4, 0],
                [-15, -6, 1]])
y = np.array([1, 0, 0, 1])
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=1000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')