"""
Модуль с функциями для распознавания рукописных цифр
"""
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Размер к которому приводить изображение
IMG_SIZE = 512

# Функция загружает с диска модель с весами на указанное устройство
def get_model(model_PATH, device_name=''):
    if device_name == '':
        model = load_model(model_PATH)
    else:
        with tf.device(device_name):
            model = load_model(model_PATH)
    return model


# Функция предикта картинки
def mnist_predict(model, img_FILE, out_FILE):
    curr_image = cv2.imread(img_FILE)
    # Переводим ее в ч/б цвет
    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    # print(curr_image.shape)
    # сохраним оригинальные размеры картинки
    curr_w = curr_image.shape[1]
    curr_h = curr_image.shape[0]
    # рассчитаем коэффициент для изменения размера
    if curr_w > curr_h:
        scale_frame = IMG_SIZE / curr_w
    else:
        scale_frame = IMG_SIZE / curr_h
    # и новые размеры изображения
    new_width = int(curr_w * scale_frame)
    new_height = int(curr_h * scale_frame)
    # делаем ресайз к целевым размерам
    curr_image = cv2.resize(curr_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Подготовим картинку для подачи в нейронку
    data = cv2.resize(curr_image, (28, 28), interpolation=cv2.INTER_AREA)
    data = np.asarray(data)
    """
    Здесь могла бы быть более сложная предобработка картинки,
    например скорректировать контраст, исправить масштаб...
    """
    data = 255 - data    # инверсия цвета
    # data = data / 255. # нормализация здесь не нужна
    image_to_pred = data.reshape(1, 28, 28, 1)
    # print(image_to_pred.shape)

    prediction = model.predict(image_to_pred)
    # print(prediction[0])
    pred = np.argmax(prediction[0])
    # print(pred)
    # Напечатаем результат работы
    print("Файл {0} распознан как {1}".format(img_FILE, pred))

    # Напишем значение pred на картинке
    cv2.putText(curr_image, "pred = " + str(pred), (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Сохраним картинку в целевую папку
    cv2.imwrite(out_FILE, curr_image)

    # имя и путь выходного файла не меняли
    return out_FILE

