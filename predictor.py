import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness



class PetClassifier:
    def __init__(self, shape=(224, 224)):
        self.shape = shape
        self.model = None

        # Слой для аугментации данных
        data_augmentation = tf.keras.Sequential([
            RandomFlip("horizontal_and_vertical"),  # Случайное отражение по горизонтали
            RandomRotation(0.2),  # Случайное вращение на ±20%
            RandomZoom(0.1),  # Случайное увеличение/уменьшение
            RandomContrast(0.2),  # Случайное изменение контраста
            RandomBrightness(0.3)
        ])

        # Загружаем предварительно обученную модель MobileNetV2 без верхнего слоя (head)
        base_model = tf.keras.applications.MobileNet(weights="imagenet",
                                                       include_top=False,
                                                       input_shape=(self.shape[0], self.shape[1], 3))
        base_model.trainable = True

        # Добавляем собственный классификатор для кошек и собак
        x = base_model.output
        x = data_augmentation(x)  # Применяем аугментацию
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation="sigmoid")(x)  # Выход для двух классов (кошки или собаки)

        # Создаем модель
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Компилируем модель
        self.model.compile(optimizer=Adam(learning_rate=1e-04), loss="binary_crossentropy", metrics=["accuracy"])


    def train(self, train_ds, val_ds, epochs=10):
        filepath = 'models/model.keras'
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )

        # Обучаем модель и сохраняем историю
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[save_callback], verbose=1)

        # Визуализация графиков loss и accuracy
        self.plot_training_history(history)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def plot_training_history(self, history):
        # Извлекаем данные из объекта истории
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Создаем график для Accuracy
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Тренировочная Accuracy')
        plt.plot(val_acc, label='Валидационная Accuracy')
        plt.title('Точность (Accuracy)')
        plt.xlabel('Эпохи')
        plt.ylabel('Точность')
        plt.legend()

        # Создаем график для Loss
        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Тренировочный Loss')
        plt.plot(val_loss, label='Валидационный Loss')
        plt.title('Потери (Loss)')
        plt.xlabel('Эпохи')
        plt.ylabel('Потери')
        plt.legend()

        # Показываем графики
        plt.tight_layout()
        plt.show()

    def predict(self, img_path):
        img = tf.keras.utils.load_img(img_path, target_size=self.shape)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        model = self.model
        probability = model.predict(img_array)[0][0]
        label = 'Кошка' if probability < 0.5 else 'Собака'
        probability = 1 - probability if label == 'Кошка' else probability
        return label, round(probability * 100, 2)