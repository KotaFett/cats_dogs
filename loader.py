import tensorflow as tf

class Loader:
    def __init__(self):
        pass

    def load_dataset(self, dir: str, batch_size: int=32, img_size: tuple=(150, 150)):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='binary',  # Для двух классов (кошки и собаки) будет 0 или 1
            seed=42
        )
        # Получите имена классов до применения map
        class_names = ds.class_names
        # Применение функции нормализации к датасету
        ds = ds.map(self.normalize_image)
        return class_names, ds

    # Функция для нормализации изображений
    @staticmethod
    def normalize_image(image, label):
        image = tf.cast(image, tf.float32)  # Преобразуем изображение в тип float32
        image = image / 255.0  # Нормализуем в диапазон [0, 1]
        return image, label


