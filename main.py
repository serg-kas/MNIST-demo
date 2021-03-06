"""
Демонстрация распознавания рукописных цифр моделью,
обученной на датасете MNIST.
При запуске обрабатывает все файлы из папки source_files.
Результат помещает в папку out_files добавляя к имени файла "pred_".
Если файл уже обработан, то его пропускает
(для повторной обработки файлов надо удалить или очистить папку  out_files).
Пути к папкам и путь к модели можно задать в аргументах.
"""
# Модуль с функциями
import mnist
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Допустимые форматы изображений
img_type_list = ['.jpg', '.jpeg', '.png']

def process(source_PATH, out_PATH, model_PATH):
    """
    source_PATH путь к каталогу с файлами
    out_PATH путь результатам
    model_PATH Путь к модели
    """
    # Создадим папки для файлов, если их нет
    if not (source_PATH in os.listdir('.')):
        os.mkdir(source_PATH)
    if not (out_PATH in os.listdir('.')):
        os.mkdir(out_PATH)

    # В папке должен быть файл модели
    assert model_PATH in os.listdir('.'), 'В папке программы должен быть файл модели'

    # Создадим список файлов для обработки
    source_files = sorted(os.listdir(source_PATH))
    out_files = sorted(os.listdir(out_PATH))
    # Список картинок для обработки
    img_files = []
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # print(f,filename,file_extension)
        if not (('out_'+f) in out_files):
            if file_extension in img_type_list:
                img_files.append(f)

    # Получаем модель
    model = mnist.get_model(model_PATH, '')
    # model = mnist.get_model(model_PATH, '/cpu:0')

    # Обрабатываем картинки
    for img in img_files:
        # полные пути к файлам
        img_FILE = source_PATH + '/' + img
        out_FILE = out_PATH + '/' + 'out_' + img
        # Вызов функции предикта
        _ = mnist.mnist_predict(model, img_FILE, out_FILE)

    # Сообщаем что обработали
    if len(img_files) == 0:
        # print('Нет картинок для обработки.')
        print('The are no pictures to predict.')
    else:
        # print('Обработали {0} картинок.'.format(len(img_files)))
        print('Predicted {0} pictures.'.format(len(img_files)))


if __name__ == '__main__':
    source_PATH = 'source_files' if len(sys.argv) <= 1 else sys.argv[1]
    out_PATH = 'out_files' if len(sys.argv) <= 2 else sys.argv[2]
    model_PATH = 'model-CNN.h5' if len(sys.argv) <= 3 else sys.argv[3]

    process(source_PATH, out_PATH, model_PATH)
