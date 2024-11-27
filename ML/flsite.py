import pickle
import os
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from model4.neuron import SingleNeuron

label_encoder=LabelEncoder()

app = Flask(__name__)



menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"},
        {"name": "Лаба 5", "url": "lab18"},
        {"name": "Лаба 6", "url": "lab19-20"}]

loaded_model_knn = pickle.load(open('model/melon', 'rb'))
loaded_model_Log = pickle.load(open('model2/melon', 'rb'))
loaded_model_Tree = pickle.load(open('model3/melon', 'rb'))
new_neuron = SingleNeuron(input_size=2)
new_neuron.load_weights('model4/neuron_weights.txt')


fashion_mnist_model = tf.keras.models.load_model('model5/fashion_mnist_model.keras')
fruit_model = tf.keras.models.load_model('model5/fruit-and-vegetable_model.keras')
fruit_model_MNV2 = tf.keras.models.load_model('model5/mobile_net_v2.keras')

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполнил студент группы ИВТ-201 Ушаков Никита Юрьевич", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[int(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3'])]])
        pred = str(loaded_model_knn.predict(X_new))
        if pred=="[0]":
            sort="Калхозница"
        elif pred=="[1]":
            sort="Торпеда"
        elif pred=="[2]":
            sort="Пятиминутка"

        elif pred=="[3]":
            sort = "Аляска"
        elif pred=="[4]":
            sort = "Сангрия"

        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + sort)

@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('kkal')),
                       float(request.args.get('width')),
                       float(request.args.get('sugar'))]])
    pred = loaded_model_knn.predict(X_new)
    print(pred[0])
    if pred == 0:
       sort = "Kalhoznitsa"
    elif pred == 1:
        sort = "Torpeda"
    elif pred == 2:
        sort = "Pyatiminutka"
    elif pred == 3:
        sort = "Alyasks"
    elif pred == 4:
        sort = "Sangria"
    else:
        sort = 'defult'
    return jsonify(sort=sort) # http://127.0.0.1:5000/api?kkal=25&width=25&sugar=3

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu)
    if request.method == 'POST':
        X_new = np.array([[int(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3'])]])
        pred = str(loaded_model_Log.predict(X_new))
        if pred == "[0]":
            sort = "Колхозница"
        elif pred == "[1]":
            sort = "Торпеда"
        elif pred == "[2]":
            sort = "Пятиминутка"

        elif pred == "[3]":
            sort = "Аляска"
        elif pred == "[4]":
            sort = "Сангрия"

        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + sort)

@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html',
                               title="Бинарное дерево",
                               menu=menu,
                               )
    if request.method == 'POST':
        x_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),]])
        pred = loaded_model_Tree.predict(x_new)
        return render_template('lab3.html',
                               title="Бинарное дерево",
                               menu=menu,
                               class_model=f"Сложность предмета: {pred[0]}")
@app.route("/p_lab4", methods=['POST', 'GET'])
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1'])-165,
                           float(request.form['list2'])-55,
                           float(request.form['list3'])]])
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Женщина', 'Мужчина'))
        return render_template('lab4.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + str(*np.where(predictions >= 0.5, 'Женщина', 'Мужчина')))

@app.route("/lab18", methods=['GET', 'POST'])
def tf_mnis():
    if request.method == 'GET':
        return render_template('lab18.html', menu=menu)

    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                raise Exception("Файл не найден")
            file = request.files['file']

            if file.filename == '':
                raise Exception("Файл не выбран")
            if file and allowed_file(file.filename):
                file_path = os.path.join('static', file.filename)
                file.save(file_path)

                img = image.load_img(file_path, target_size=(28, 28), color_mode='grayscale')
                x = image.img_to_array(img)
                x = x.reshape(1, 784)  # Нормализация для модели
                x /= 255.0  # Нормализация данных

                prediction = fashion_mnist_model.predict(x)  # Предсказание на основе загруженного изображения
                predicted_class_index = np.argmax(prediction)

                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                predicted_class_name = class_names[predicted_class_index]

                return render_template('lab18.html', class_name=predicted_class_name, image_path=file.filename, menu=menu,)

            else:
                raise Exception("Неподдерживаемый файл")
        except Exception as e:
            print("Ошибка:", e)
            return render_template('lab18.html', error=str(e), menu=menu,)


# Функция для загрузки названий классов из файла
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Загружаем названия классов один раз при старте приложения
class_names = load_class_names('model5/class_names.txt')

@app.route("/lab19-20", methods=['GET', 'POST'])
def my_mnis():
    if request.method == 'GET':
        return render_template('lab19-20.html', menu=menu,)

    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                raise Exception("Файл не найден")
            file = request.files['file']
            if file.filename == '':
                raise Exception("Файл не выбран")

            if file and allowed_file(file.filename):
                file_path = os.path.join('static', file.filename)
                file.save(file_path)

                # Загрузка и предварительная обработка изображения
                img = image.load_img(file_path, target_size=(180, 180))  # Корректный размер
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)  # Создаем пакет из 1 изображения
                x /= 255.0  # Нормализация

                # Получаем выбранную модель
                selected_model = request.form['model']

                # Предсказание с использованием выбранной модели
                if selected_model == 'model1':
                    prediction = fruit_model_MNV2.predict(x)  # Используем MobileNetV2
                    model_name = "Предобученная сеть"
                elif selected_model == 'model2':
                    prediction = fruit_model.predict(x)  # Используем собственный нейрон
                    model_name = "Собственная сеть"

                # Определяем класс и вероятность из предсказания
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                prediction_probability = np.max(prediction)  # Максимальная вероятность

                # Выводим информацию в консоль
                print(f"Модель: {model_name}, Предсказанный класс: {predicted_class_name}, Вероятность: {prediction_probability:.4f}")

                return render_template('lab19-20.html', class_name=predicted_class_name, image_path=file.filename, menu=menu,)

            else:
                raise Exception("Неподдерживаемый файл")
        except Exception as e:
            print("Ошибка:", e)
            return render_template('lab19-20.html', error=str(e), menu=menu)



# Функция для проверки типов файлов
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
if __name__ == "__main__":
    app.run(debug=True)
