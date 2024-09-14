import os
import json

# Путь к файлам
LABEL_FILE_PATH = 'label.json'  # Файл с метками изображений
# TARGET_FOLDER = 'data/anatomy/z-line'  # Папка с изображениями для разметки
TARGET_FOLDER = 'data/pathology/normal'
# TARGET_FOLDER = 'data/pathology/barrett'
# TARGET_FOLDER = 'data/pathology/esophagitis-a'
# TARGET_FOLDER = 'data/pathology/esophagitis-b-d'
# TARGET_FOLDER = 'data/anatomy/retroflex-stomach'
# TARGET_FOLDER = 'data/anatomy/retroflex-rectum'



# Загрузка или создание label.json
def load_label_file():
    if os.path.exists(LABEL_FILE_PATH):
        with open(LABEL_FILE_PATH, 'r') as f:
            label_data = json.load(f)
    else:
        label_data = {}  # Если файл не существует, начинаем с пустого словаря
    return label_data

# Сохранение данных в label.json
def save_label_file(label_data):
    with open(LABEL_FILE_PATH, 'w') as f:
        json.dump(label_data, f, indent=4)

# Функция для обработки файлов в указанной папке
def process_folder(folder_path):
    # Загружаем или создаем label.json
    label_data = load_label_file()
    
    # Получаем категорию (например, anatomy) и метку (например, z-line) из пути папки
    category_name = os.path.basename(os.path.dirname(folder_path))  # Например, "anatomy"
    label_name = os.path.basename(folder_path)  # Например, "z-line"
    
    # Перебираем все файлы в папке
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            continue  # Пропускаем, если это не файл

        # Если файл уже есть в label.json, обновляем его метки
        if filename in label_data:
            label_data[filename][category_name] = label_name
        else:
            # Добавляем новую запись для файла
            label_data[filename] = {
                category_name: label_name
            }
    
    # Сохраняем обновленный label.json
    save_label_file(label_data)

# Основная функция для запуска
def main():
    # Обрабатываем указанную папку
    process_folder(TARGET_FOLDER)

if __name__ == '__main__':
    main()
