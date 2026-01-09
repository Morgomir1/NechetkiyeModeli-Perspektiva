# -*- coding: utf-8 -*-
import sys
import io
import pickle
import numpy as np

# Настройка кодировки для Windows консоли
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("Проверка данных для генерации графиков")
print("=" * 60)

# Проверка данных ANFIS
print("\n1. Проверка anfis_results.pkl:")
try:
    with open('anfis_results.pkl', 'rb') as f:
        anfis_data = pickle.load(f)
    print(f"   ✓ Файл загружен")
    print(f"   Ключи: {list(anfis_data.keys())}")
    
    # Проверка истории
    if 'history' in anfis_data:
        history = anfis_data['history']
        print(f"   history: {type(history)}")
        if history and isinstance(history, dict):
            print(f"   history keys: {list(history.keys())}")
            if 'loss' in history:
                loss = history['loss']
                print(f"   loss type: {type(loss)}, length: {len(loss) if hasattr(loss, '__len__') else 'N/A'}")
                if len(loss) > 0:
                    print(f"   loss[0]: {loss[0] if hasattr(loss, '__getitem__') else 'N/A'}")
        else:
            print(f"   history value: {history}")
    else:
        print("   ✗ history отсутствует")
    
    # Проверка предсказаний
    for key in ['y_train_pred', 'y_test_pred', 'y_train_orig', 'y_test_orig']:
        if key in anfis_data:
            val = anfis_data[key]
            if val is not None:
                arr = np.asarray(val)
                print(f"   {key}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")
            else:
                print(f"   {key}: None")
        else:
            print(f"   ✗ {key} отсутствует")
            
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# Проверка изображений
print("\n2. Проверка plot_images.pkl:")
try:
    with open('plot_images.pkl', 'rb') as f:
        images = pickle.load(f)
    print(f"   ✓ Файл загружен")
    print(f"   Ключи: {list(images.keys())}")
    for key in ['training_history', 'train_predictions', 'test_predictions', 'train_residuals', 'test_residuals']:
        if key in images:
            val = images[key]
            if val:
                print(f"   {key}: присутствует (длина: {len(val)} символов)")
            else:
                print(f"   {key}: пустое")
        else:
            print(f"   ✗ {key} отсутствует")
except FileNotFoundError:
    print("   ✗ Файл не найден")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Попытка генерации графиков...")
print("=" * 60)

# Попытка генерации графиков
try:
    from generate_plots import generate_all_plots
    images = generate_all_plots()
    print(f"\nРезультат: сгенерировано {len(images)} графиков")
    print(f"Ключи: {list(images.keys())}")
except Exception as e:
    print(f"\n✗ Ошибка при генерации: {e}")
    import traceback
    traceback.print_exc()
