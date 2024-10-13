import os
import json
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from audio_processing import clean_and_amplify_audio
from speech_recognition import load_commands_from_json, transcribe_audio
import vosk

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # Возвращает использование памяти в МБ

def process_single_file(file_name, folder_path, model, commands):
    input_file = os.path.join(folder_path, file_name)
    start_time = time.time()
    
    recognized_text, command_id = transcribe_audio(input_file, model, commands)
    elapsed_time = time.time() - start_time
    return file_name, recognized_text, command_id, elapsed_time

def process_folder(folder_path, model, commands, log_file):
    log_data = []
    total_elapsed_time = 0
    peak_memory_usage = 0  # Для отслеживания пикового использования памяти

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    with ThreadPoolExecutor() as executor:
        futures = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                futures.append(executor.submit(process_single_file, file_name, folder_path, model, commands))

        for future in futures:
            file_name, recognized_text, command_id, elapsed_time = future.result()
            log_data.append({"file": file_name, "recognized_text": recognized_text, "command_id": command_id, "elapsed_time": elapsed_time})
            print(f"Recognized: '{recognized_text}', Command ID: {command_id}, Time: {elapsed_time:.2f} seconds")

            # Измеряем использование памяти после обработки каждого файла
            current_memory_usage = get_memory_usage()
            peak_memory_usage = max(peak_memory_usage, current_memory_usage)

    total_elapsed_time += elapsed_time

    print(f"\nTotal time for all files: {total_elapsed_time:.2f} seconds")
    print(f"Peak memory usage: {peak_memory_usage:.2f} MB")

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    commands_json_file = 'commands.json'
    commands = load_commands_from_json(commands_json_file)
    folder_path = 'audio_files'
    model_path = "model_small"
    
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
    else:
        model = vosk.Model(model_path)
        log_file = 'log.json'
        
        # Измеряем использование памяти
        process_folder(folder_path, model, commands, log_file)
