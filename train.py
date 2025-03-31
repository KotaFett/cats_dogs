from predictor import PetClassifier
from loader import Loader

# Основной блок для запуска обучения
if __name__ == "__main__":
    # Загружаем данные
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'
    shape = (224, 224)
    batch_size = 16
    loader = Loader()
    train_class_names, train_ds = loader.load_dataset(train_dir, img_size=shape, batch_size=batch_size)
    val_class_names, val_ds = loader.load_dataset(val_dir, img_size=shape, batch_size=batch_size)
    test_class_names, test_ds = loader.load_dataset(test_dir, img_size=shape, batch_size=batch_size)
    # Создаем экземпляр классификатора
    classifier = PetClassifier(shape=shape)

    # Обучаем модель
    classifier.train(train_ds, val_ds, epochs=10)

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
    print(f"Предсказание dog_2.jpg: {label} с вероятностью {probability}%")

    # Пример предсказания
    img_path = 'uploads/human_1.jpg'  # Укажите путь к изображению для предсказания
    label, probability = classifier.predict(img_path)
    print(f"Предсказан human_1.jpg: {label} с вероятностью {probability}%")