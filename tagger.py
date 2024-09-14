import os
import json

# Путь к файлам
LABELS_DICT_PATH = 'labels_dict.json'  # Словарь с метками
LABEL_FILE_PATH = 'label.json'  # Файл с метками изображений
TARGET_FOLDER = 'data/anatomy/z-line'  # Папка с изображениями для разметки

# Загрузка labels_dict.json
def load_labels_dict():
    with open(LABELS_DICT_PATH, 'r') as f:
        labels_dict = json.load(f)
    return labels_dict[0]  # Извлекаем первый элемент из списка

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
def process_folder(folder_path, labels_dict):
    # Загружаем или создаем label.json
    label_data = load_label_file()
    
    # Получаем анатомическую метку из пути папки (например, oesophagus)
    anatomy_name = os.path.basename(folder_path)
    anatomy_label = labels_dict['anatomy'].get(anatomy_name)
    
    if anatomy_label is None:
        print(f"Warning: Anatomy label not found for {anatomy_name}")
        return
    
    # Перебираем все файлы в папке
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            continue  # Пропускаем, если это не файл

        # Если файл уже есть в label.json, обновляем его метки
        if filename in label_data:
            label_data[filename]['anatomy'] = anatomy_label
        else:
            # Запрашиваем метки качества и патологии у пользователя
            print(f"Processing {filename}:")
            quality_label = get_label_input('quality', labels_dict)
            pathology_label = get_label_input('pathology', labels_dict)
            
            # Создаем новую запись для файла
            label_data[filename] = {
                'anatomy': anatomy_label,
                'quality': quality_label,
                'pathology': pathology_label
            }
    
    # Сохраняем обновленный label.json
    save_label_file(label_data)

# Запрашиваем метку от пользователя для качества и патологии
def get_label_input(label_type, labels_dict):
    label_options = labels_dict[label_type]
    print(f"Available {label_type} options:")
    for key, value in label_options.items():
        print(f"{key}: {value}")
    
    while True:
        user_input = input(f"Enter {label_type} label: ").strip().lower()
        if user_input in label_options:
            return label_options[user_input]
        else:
            print(f"Invalid {label_type} label. Please try again.")

# Основная функция для запуска
def main():
    # Загружаем словарь меток
    labels_dict = load_labels_dict()
    
    # Обрабатываем папку
    process_folder(TARGET_FOLDER, labels_dict)

if __name__ == '__main__':
    main()
