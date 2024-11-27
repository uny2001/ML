from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Преобразование данных
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

# Преобразование меток в категориальный формат
nb_classes = 10
Y_train = utils.to_categorical(y_train, nb_classes)
Y_test = utils.to_categorical(y_test, nb_classes)

# Создание модели
model = Sequential()
model.add(Dense(512, input_shape=(784,), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

model.summary()

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=1)

# Оценка качества обучения
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress',
    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

scores = model.evaluate(X_test, Y_test, verbose=1)
print(f'Значение функции потерь (loss) на тестовых данных: {scores[0]}')
print(f'Доля верных ответов на тестовых данных, в процентах (accuracy): {round(scores[1] * 100, 4)}')

# Сохранение модели
model.save('fashion_mnist_model.keras')  # Сохраните модель в файл формата HDF5

print("Модель сохранена в файл fashion_mnist_model.keras")