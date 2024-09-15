import os
import uuid

# Папка с изображениями
folder = 'temp/retro-rectum'

# Проходим по всем файлам в папке с изображениями
for filename in os.listdir(folder):
    # Проверяем, является ли файл изображением (по расширению)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Получаем путь к исходному файлу
        file_path = os.path.join(folder, filename)

        # Извлекаем расширение файла
        file_extension = os.path.splitext(filename)[1]

        # Генерируем UUID для нового имени файла
        new_filename = f"{uuid.uuid4()}{file_extension}"

        # Путь для сохранения с новым именем
        new_file_path = os.path.join(folder, new_filename)

        # Переименовываем файл (заменяем старый файл на новый с UUID)
        os.rename(file_path, os.path.join(folder, new_filename))

        print(f"Файл {filename} переименован в {new_filename}")
