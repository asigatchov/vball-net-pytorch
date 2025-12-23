# convert_v2_to_v1b.py

import torch
import os
import argparse
import re


def convert_v2_checkpoint_to_v1b(input_path: str, output_path: str = None):
    """
    Конвертирует чекпоинт VballNetV2 в формат, совместимый с VballNetV1b.
    Меняет префикс ключей в state_dict: 'VballNetV2.' → 'VballNetV1b.'

    Параметры:
        input_path (str): путь к исходному .pth файлу (VballNetV2)
        output_path (str): путь для сохранения нового файла. 
                           Если None — добавит суффикс '_v1b' к исходному имени.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    print(f"Загрузка чекпоинта: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Определяем state_dict в зависимости от структуры чекпоинта
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            checkpoint_type = 'with_wrapper'
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            checkpoint_type = 'with_wrapper'
        else:
            state_dict = checkpoint
            checkpoint_type = 'raw'
    else:
        state_dict = checkpoint
        checkpoint_type = 'raw'

    print(f"Тип чекпоинта: {checkpoint_type}")
    print(f"Количество параметров до конвертации: {len(state_dict)}")

    # Подсчёт и замена префиксов
    converted_count = 0
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('VballNetV2.'):
            new_key = 'VballNetV1b.' + key[len('VballNetV2.'):]
            converted_count += 1
        else:
            new_key = key  # оставляем без изменений (например, optimizer, epoch и т.д.)
        new_state_dict[new_key] = value

    print(f"Заменено префиксов: {converted_count}")

    # Восстанавливаем структуру чекпоинта
    if checkpoint_type == 'with_wrapper':
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = new_state_dict
        elif 'model_state_dict' in checkpoint:
            checkpoint['model_state_dict'] = new_state_dict
        converted_checkpoint = checkpoint
    else:
        converted_checkpoint = new_state_dict

    # Определяем путь сохранения
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        # Автоматически определяем seq, если есть в имени
        seq_match = re.search(r'seq(\d+)', base)
        seq_suffix = f"_seq{seq_match.group(1)}" if seq_match else ""
        output_path = f"{base}{seq_suffix}_v1b{ext}"

    print(f"Сохранение конвертированного чекпоинта: {output_path}")
    torch.save(converted_checkpoint, output_path)
    print("Конвертация завершена успешно!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Конвертация чекпоинта VballNetV2 → VballNetV1b (переименование класса в state_dict)"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Путь к исходному .pth файлу (VballNetV2)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Путь для сохранения нового файла. Если не указан — добавится суффикс '_v1b'"
    )

    args = parser.parse_args()

    convert_v2_checkpoint_to_v1b(args.input, args.output)
