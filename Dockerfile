# Используем официальный образ TensorFlow
FROM tensorflow/tensorflow:2.17.0

# Установим рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app/
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --upgrade pip
# Удаляем старую версию blinker, если она есть
RUN python -m pip uninstall -y blinker || true
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем приложение
CMD ["python", "app.py"]
