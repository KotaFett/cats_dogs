from flask import Flask, request, render_template, jsonify
from predictor import PetClassifier
import os
from flask import send_from_directory


app = Flask(__name__)

# Инициализируем классификатор
classifier = PetClassifier()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "image" in request.files:
        img_file = request.files["image"]

        # Сохраняем изображение
        img_path = os.path.join("uploads", img_file.filename)
        img_file.save(img_path)

        # Получаем предсказание
        label, probability = classifier.predict(img_path)
        return render_template("result.html",
                               label=label,
                               probability=probability,
                               image_path=img_path)

    return render_template("index.html")


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


if __name__ == "__main__":
    # Запускаем сервер
    app.run(debug=False, use_reloader=True, host="0.0.0.0", port=80)

