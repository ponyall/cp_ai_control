import json
import difflib
import os
import vosk

def load_commands_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_best_match(input_text: str, labels) -> tuple:
    best_match = None
    best_ratio = 0.0
    for label in labels:
        match_ratio = difflib.SequenceMatcher(None, input_text, label).ratio()
        if match_ratio > best_ratio:
            best_ratio = match_ratio
            best_match = label
    return best_match, best_ratio

def transcribe_audio(file_path, model, commands):
    rec = vosk.KaldiRecognizer(model, 16000)

    with open(file_path, "rb") as f:
        wf = f.read()
        rec.AcceptWaveform(wf)
        result = rec.Result()

        result_dict = json.loads(result)
        recognized_text = result_dict.get('text', '')

        best_match, _ = find_best_match(recognized_text, commands)
        command_id = commands.get(best_match, None)  # Получаем ID команды

        return recognized_text, command_id
