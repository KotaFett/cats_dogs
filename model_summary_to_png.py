import tensorflow as tf
import matplotlib.pyplot as plt

# Загружаем модель
model = tf.keras.models.load_model('models/model.keras')

# Получаем саммари модели
summary = []
model.summary(print_fn=lambda x: summary.append(x))

# Настроим отображение текста
plt.figure(figsize=(12, 10))
plt.text(0.01, 1, '\n'.join(summary), {'fontsize': 10}, fontfamily='monospace')

# Отключаем оси
plt.axis('off')

# Сохраняем изображение
plt.savefig('model_summary.png', bbox_inches='tight', pad_inches=0.1)
plt.close()
