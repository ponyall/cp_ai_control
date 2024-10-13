import os
import json
from audio_processing import clean_and_amplify_audio
from speech_recognition import load_commands_from_json, transcribe_audio
import vosk

class Predictor:
    """Класс для предсказаний модели."""
    def __init__(self, model_path, commands):
        self.model = vosk.Model(model_path)
        self.commands = commands

    def __call__(self, audio_path: str):
        recognized_text, command_id = transcribe_audio(audio_path, self.model, self.commands)
        result = {
            "audio": os.path.basename(audio_path), # Имя аудиофайла
            "text": recognized_text, # Распознанный текст
            "label": command_id # ID команды
        }
        return result

def main():
    # Запрашиваем пути
    src = input("Введите путь к папке с аудиофайлами: ")
    dst = input("Введите путь к папке для сохранения submission.json: ")

    # Проверяем существование указанных путей
    if not os.path.exists(src):
        print(f"Папка {src} не существует.")
        return

    if not os.path.exists(dst):
        os.makedirs(dst)

    # Загружаем команды и создаем объект Predictor
    commands = load_commands_from_json("commands.json")
    predictor = Predictor("model_small", commands)

    # Список для результатов
    results = []

    # Обрабатываем каждый файл в папке
    for file_name in os.listdir(src):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(src, file_name)
            result = predictor(audio_path)
            results.append(result)

    # Сохраняем результаты в файл submission.json
    output_file = os.path.join(dst, "submission.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Результаты сохранены в {output_file}")

if __name__ == "__main__":
    main()
