import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from sklearn.preprocessing import LabelEncoder
from model4.neuron import SingleNeuron

label_encoder=LabelEncoder()

app = Flask(__name__)



menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"}]

loaded_model_knn = pickle.load(open('model/melon', 'rb'))
loaded_model_Log = pickle.load(open('model2/melon', 'rb'))
loaded_model_Tree = pickle.load(open('model3/melon', 'rb'))
new_neuron = SingleNeuron(input_size=2)
new_neuron.load_weights('model4/neuron_weights.txt')

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
if __name__ == "__main__":
    app.run(debug=True)
