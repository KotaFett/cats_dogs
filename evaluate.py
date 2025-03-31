import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loader import Loader
from predictor import PetClassifier


def evaluate_model(model, test_ds, test_class_names):
    """
    Оценка модели с использованием метрик accuracy, precision, recall, confusion matrix.
    """

    for images, labels in test_ds.take(1):
        print(f"Форма изображений: {images.shape}")
        print(f"Форма меток: {labels.shape}")

    # Генерация предсказаний для тестового набора данных
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        predictions = model.predict(images)
        # Преобразуем вероятности в бинарные предсказания (0 или 1)
        binary_predictions = (predictions >= 0.5).astype(int)  # Если вероятность >= 0.5, то 1, иначе 0

        # print(predictions)
    #     predictions = model.evaluate(test_ds)
        y_true.extend(labels.numpy())
        y_pred.extend(binary_predictions)


    # Перевод в numpy массивы
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(y_true.shape)
    print(y_pred.shape)
    # Расчет метрик
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    # Вывод результатов
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    # Визуализация матрицы ошибок
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_class_names, yticklabels=test_class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


if __name__ == "__main__":
    # Загружаем данные
    test_dir = 'dataset/test'
    shape = (224, 224)
    batch_size = 16
    loader = Loader()
    test_class_names, test_ds = loader.load_dataset(test_dir, img_size=shape, batch_size=batch_size)

    # Укажите путь к обученной модели
    model_path = 'models/model.keras'
    classifier = PetClassifier()
    classifier.load_model(model_path)
    # Оценка модели
    evaluate_model(classifier.model, test_ds, test_class_names)

    # Пример предсказания
    img_path = 'uploads/cat_1.jpg'  # Укажите путь к изображению для предсказания
    label, probability = classifier.predict(img_path)
    print(f"Предсказание cat_1.jpg: {label} с вероятностью {probability}%")

    # Пример предсказания
    img_path = 'uploads/cat_2.jpg'  # Укажите путь к изображению для предсказания
    label, probability = classifier.predict(img_path)
    print(f"Предсказание cat_2.jpg: {label} с вероятностью {probability}%")

    # Пример предсказания
    img_path = 'uploads/dog_1.jpg'  # Укажите путь к изображению для предсказания
    label, probability = classifier.predict(img_path)
    print(f"Предсказание dog_1.jpg: {label} с вероятностью {probability}%")

    # Пример предсказания
    img_path = 'uploads/dog_2.jpg'  # Укажите путь к изображению для предсказания
    label, probability = classifier.predict(img_path)
    print(f"Предсказан dog_2.jpg: {label} с вероятностью {probability}%")

    # Пример предсказания
    img_path = 'uploads/human_1.jpg'  # Укажите путь к изображению для предсказания
    label, probability = classifier.predict(img_path)
    print(f"Предсказан human_1.jpg: {label} с вероятностью {probability}%")
