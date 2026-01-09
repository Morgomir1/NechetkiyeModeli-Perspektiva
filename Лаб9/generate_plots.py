# -*- coding: utf-8 -*-
import sys
import io
import os

# Настройка кодировки для Windows консоли (более безопасный способ)
if sys.platform == 'win32':
    # Устанавливаем переменную окружения для кодировки
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Перенастраиваем только если это не subprocess
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass
    if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass

import numpy as np
import matplotlib.pyplot as plt
import pickle
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def fig_to_base64(fig):
    """Конвертирует matplotlib figure в base64 строку"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_membership_functions(system, variable_name, title):
    """Построение графиков функций принадлежности"""
    if variable_name not in system.input_variables:
        return None
    
    var_data = system.input_variables[variable_name]
    universe = var_data['universe']
    membership_functions = var_data['membership_functions']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (term, mf_values) in enumerate(membership_functions.items()):
        color = colors[idx % len(colors)]
        ax.plot(universe, mf_values, label=term, linewidth=2, color=color)
    
    ax.set_xlabel(variable_name, fontsize=12)
    ax.set_ylabel('Степень принадлежности', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    return fig_to_base64(fig)


def plot_surface_3d(system, var1_name, var2_name, var1_range, var2_range, title):
    """Построение 3D поверхности выхода системы"""
    from mpl_toolkits.mplot3d import Axes3D
    
    var1_vals = np.linspace(var1_range[0], var1_range[1], 30)
    var2_vals = np.linspace(var2_range[0], var2_range[1], 30)
    var1_grid, var2_grid = np.meshgrid(var1_vals, var2_vals)
    
    output_grid = np.zeros_like(var1_grid)
    
    for i in range(len(var1_vals)):
        for j in range(len(var2_vals)):
            inputs = {var1_name: var1_vals[i], var2_name: var2_vals[j]}
            output_grid[j, i] = system.infer(inputs)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(var1_grid, var2_grid, output_grid, 
                          cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    
    ax.set_xlabel(var1_name, fontsize=11)
    ax.set_ylabel(var2_name, fontsize=11)
    ax.set_zlabel('Выход системы', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig_to_base64(fig)


def plot_anfis_training_history(history):
    """Построение графика истории обучения ANFIS"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Преобразование в список, если это numpy массив
    loss = history.get('loss', [])
    if hasattr(loss, 'tolist'):
        loss = loss.tolist()
    elif not isinstance(loss, list):
        loss = list(loss)
    
    if len(loss) == 0:
        # Если история пуста, создаем пустой график
        ax.text(0.5, 0.5, 'История обучения пуста', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
    else:
        epochs = range(1, len(loss) + 1)
        loss_array = np.array(loss)
        
        # Проверяем, есть ли изменение в значениях
        loss_min = loss_array.min()
        loss_max = loss_array.max()
        loss_range = loss_max - loss_min
        
        # Если изменения очень малы, используем относительную шкалу
        if loss_range < 1e-6:
            # Все значения почти одинаковые - показываем абсолютные значения
            ax.plot(epochs, loss, 'b-', linewidth=2, label=f'Loss (≈{loss_min:.6f})', marker='o', markersize=3)
            ax.set_ylabel('Ошибка (MSE)', fontsize=12)
            # Устанавливаем небольшой диапазон вокруг среднего значения
            loss_mean = loss_array.mean()
            ax.set_ylim([loss_mean - max(1e-6, loss_range * 10), loss_mean + max(1e-6, loss_range * 10)])
        else:
            # Есть заметные изменения - обычный график
            ax.plot(epochs, loss, 'b-', linewidth=2, label='Loss', marker='o', markersize=2)
            ax.set_ylabel('Ошибка (MSE)', fontsize=12)
            # Автоматический масштаб
            ax.set_ylim([loss_min - loss_range * 0.1, loss_max + loss_range * 0.1])
        
        # Добавляем информацию о начальном и конечном значении
        ax.text(0.02, 0.98, f'Начало: {loss[0]:.6f}\nКонец: {loss[-1]:.6f}\nИзменение: {loss[0]-loss[-1]:.6f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Эпоха', fontsize=12)
    ax.set_title('История обучения ANFIS', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def plot_predictions_vs_actual(y_true, y_pred, title):
    """Построение графика предсказаний vs фактических значений"""
    # Преобразование в numpy массивы
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, color='#2ca02c')
    
    # Линия идеального предсказания
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное предсказание')
    
    ax.set_xlabel('Фактические значения', fontsize=12)
    ax.set_ylabel('Предсказанные значения', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def plot_residuals(y_true, y_pred, title):
    """Построение графика остатков"""
    # Преобразование в numpy массивы
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График остатков
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50, color='#d62728')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xlabel('Предсказанные значения', fontsize=12)
    ax1.set_ylabel('Остатки', fontsize=12)
    ax1.set_title('График остатков', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Гистограмма остатков
    ax2.hist(residuals, bins=20, color='#9467bd', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Остатки', fontsize=12)
    ax2.set_ylabel('Частота', fontsize=12)
    ax2.set_title('Распределение остатков', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    return fig_to_base64(fig)


def generate_all_plots():
    """Генерация всех графиков для отчета"""
    images = {}
    
    # Загрузка результатов системы Сугено
    try:
        with open('sugeno_credit_results.pkl', 'rb') as f:
            sugeno_data = pickle.load(f)
        
        # Создаем систему заново (т.к. она не сохраняется из-за функций)
        from sugeno_credit_system import create_credit_sugeno_system
        system = create_credit_sugeno_system()
        
        # Графики функций принадлежности
        print("Построение графиков функций принадлежности...")
        images['age_membership'] = plot_membership_functions(
            system, 'возраст', 'Функции принадлежности для переменной "Возраст"'
        )
        images['income_membership'] = plot_membership_functions(
            system, 'доход', 'Функции принадлежности для переменной "Доход"'
        )
        
        # 3D поверхность
        print("Построение 3D поверхности...")
        images['surface'] = plot_surface_3d(
            system, 'возраст', 'доход', 
            [18, 70], [20000, 150000],
            'Поверхность выхода системы Такаги-Сугено'
        )
    except Exception as e:
        print(f"Ошибка при загрузке данных Сугено: {e}")
    
    # Загрузка результатов ANFIS
    try:
        print("\n" + "="*60)
        print("Загрузка данных ANFIS для генерации графиков")
        print("="*60)
        
        # Импортируем класс ANFIS перед загрузкой pickle, чтобы он был доступен
        import sys
        import importlib.util
        
        # Загружаем модуль anfis_training для доступа к классу ANFIS
        try:
            # Пытаемся импортировать напрямую
            from anfis_training import ANFIS
            # Регистрируем в sys.modules для pickle
            if 'anfis_training' not in sys.modules:
                import anfis_training
                sys.modules['anfis_training'] = anfis_training
        except ImportError:
            # Если прямой импорт не работает, загружаем через importlib
            spec = importlib.util.spec_from_file_location("anfis_training", "anfis_training.py")
            anfis_module = importlib.util.module_from_spec(spec)
            sys.modules['anfis_training'] = anfis_module
            spec.loader.exec_module(anfis_module)
            ANFIS = anfis_module.ANFIS
        
        # Также регистрируем в __main__ для совместимости с pickle
        if '__main__' in sys.modules:
            sys.modules['__main__'].ANFIS = ANFIS
        
        # Теперь загружаем pickle файл
        with open('anfis_results.pkl', 'rb') as f:
            anfis_data = pickle.load(f)
        
        print(f"Загружены данные ANFIS. Ключи: {list(anfis_data.keys())}")
        
        # Проверяем наличие всех необходимых ключей
        required_keys = ['y_train_pred', 'y_test_pred', 'y_train_orig', 'y_test_orig', 'history']
        missing_keys = [key for key in required_keys if key not in anfis_data]
        if missing_keys:
            print(f"  [WARN] Отсутствующие ключи: {missing_keys}")
        else:
            print(f"  [OK] Все необходимые ключи присутствуют")
        
        # Детальная проверка каждого ключа
        for key in required_keys:
            if key in anfis_data:
                val = anfis_data[key]
                if val is None:
                    print(f"  [WARN] {key}: присутствует, но равен None")
                elif key == 'history':
                    if isinstance(val, dict) and 'loss' in val:
                        loss_len = len(val['loss']) if hasattr(val['loss'], '__len__') else 0
                        print(f"  [OK] {key}: присутствует, тип={type(val).__name__}, loss length={loss_len}")
                    else:
                        print(f"  [WARN] {key}: присутствует, но некорректный формат. Тип={type(val).__name__}")
                else:
                    arr = np.asarray(val)
                    print(f"  [OK] {key}: присутствует, shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"  [ERROR] {key}: отсутствует!")
        
        # График истории обучения
        print("\n--- Генерация графика истории обучения ---")
        if 'history' in anfis_data:
            history = anfis_data['history']
            print(f"  history type: {type(history)}")
            print(f"  history value: {history}")
            
            if history is None:
                print("  [ERROR] history равен None")
            elif not isinstance(history, dict):
                print(f"  [ERROR] history не является словарем. Тип: {type(history)}")
            elif 'loss' not in history:
                print(f"  [ERROR] В history отсутствует ключ 'loss'. Ключи: {list(history.keys())}")
            elif not history['loss']:
                print(f"  [ERROR] history['loss'] пуст. Тип: {type(history['loss'])}, Длина: {len(history['loss']) if hasattr(history['loss'], '__len__') else 'N/A'}")
            else:
                loss_len = len(history['loss']) if hasattr(history['loss'], '__len__') else 0
                print(f"  [OK] history['loss'] присутствует, длина: {loss_len}")
                print("Построение графика истории обучения...")
                try:
                    result = plot_anfis_training_history(history)
                    if result:
                        images['training_history'] = result
                        print(f"  [OK] График истории обучения создан и сохранен (длина base64: {len(result)} символов)")
                    else:
                        print("  [ERROR] plot_anfis_training_history вернул None или пустую строку")
                except Exception as e:
                    print(f"  [ERROR] Ошибка при построении графика истории обучения: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("  [ERROR] Ключ 'history' отсутствует в anfis_data")
        
        # Графики предсказаний
        print("\n--- Генерация графиков предсказаний ---")
        if 'y_train_pred' in anfis_data and anfis_data['y_train_pred'] is not None:
            try:
                y_train_orig = anfis_data.get('y_train_orig')
                if y_train_orig is None:
                    y_train_orig = anfis_data.get('y_train')
                
                if y_train_orig is not None:
                    # Преобразуем в массивы для проверки
                    y_train_orig = np.asarray(y_train_orig)
                    y_train_pred = np.asarray(anfis_data['y_train_pred'])
                    
                    if len(y_train_orig) > 0 and len(y_train_pred) > 0:
                        print(f"  [OK] Данные готовы: y_train_orig shape={y_train_orig.shape}, y_train_pred shape={y_train_pred.shape}")
                        try:
                            pred_result = plot_predictions_vs_actual(
                                y_train_orig, y_train_pred,
                                'Предсказания на обучающей выборке'
                            )
                            if pred_result:
                                images['train_predictions'] = pred_result
                                print(f"  [OK] График train_predictions создан (длина: {len(pred_result)} символов)")
                            else:
                                print("  [ERROR] plot_predictions_vs_actual вернул None")
                            
                            res_result = plot_residuals(
                                y_train_orig, y_train_pred,
                                'Остатки на обучающей выборке'
                            )
                            if res_result:
                                images['train_residuals'] = res_result
                                print(f"  [OK] График train_residuals создан (длина: {len(res_result)} символов)")
                            else:
                                print("  [ERROR] plot_residuals вернул None")
                        except Exception as e:
                            print(f"  [ERROR] Ошибка при создании графиков: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"  [WARN] Данные обучающей выборки пусты: y_train_orig={len(y_train_orig)}, y_train_pred={len(y_train_pred)}")
                else:
                    print("  [WARN] y_train_orig отсутствует в данных")
            except Exception as e:
                print(f"  [ERROR] Ошибка при построении графиков обучающей выборки: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [WARN] y_train_pred отсутствует или равен None в данных")
        
        if 'y_test_pred' in anfis_data and anfis_data['y_test_pred'] is not None:
            try:
                y_test_orig = anfis_data.get('y_test_orig')
                if y_test_orig is None:
                    y_test_orig = anfis_data.get('y_test')
                
                if y_test_orig is not None:
                    # Преобразуем в массивы для проверки
                    y_test_orig = np.asarray(y_test_orig)
                    y_test_pred = np.asarray(anfis_data['y_test_pred'])
                    
                    if len(y_test_orig) > 0 and len(y_test_pred) > 0:
                        print(f"  [OK] Данные готовы: y_test_orig shape={y_test_orig.shape}, y_test_pred shape={y_test_pred.shape}")
                        try:
                            pred_result = plot_predictions_vs_actual(
                                y_test_orig, y_test_pred,
                                'Предсказания на тестовой выборке'
                            )
                            if pred_result:
                                images['test_predictions'] = pred_result
                                print(f"  [OK] График test_predictions создан (длина: {len(pred_result)} символов)")
                            else:
                                print("  [ERROR] plot_predictions_vs_actual вернул None")
                            
                            res_result = plot_residuals(
                                y_test_orig, y_test_pred,
                                'Остатки на тестовой выборке'
                            )
                            if res_result:
                                images['test_residuals'] = res_result
                                print(f"  [OK] График test_residuals создан (длина: {len(res_result)} символов)")
                            else:
                                print("  [ERROR] plot_residuals вернул None")
                        except Exception as e:
                            print(f"  [ERROR] Ошибка при создании графиков: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"  [WARN] Данные тестовой выборки пусты: y_test_orig={len(y_test_orig)}, y_test_pred={len(y_test_pred)}")
                else:
                    print("  [WARN] y_test_orig отсутствует в данных")
            except Exception as e:
                print(f"  [ERROR] Ошибка при построении графиков тестовой выборки: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [WARN] y_test_pred отсутствует или равен None в данных")
    except Exception as e:
        print(f"Ошибка при загрузке данных ANFIS: {e}")
        import traceback
        traceback.print_exc()
    
    # Сохранение изображений
    print(f"\nИтоговый список графиков:")
    for key, value in images.items():
        if value:
            print(f"  [OK] {key}: создан (длина base64: {len(value)} символов)")
        else:
            print(f"  [X] {key}: пустое значение")
    
    with open('plot_images.pkl', 'wb') as f:
        pickle.dump(images, f)
    
    print(f"\nСгенерировано {len(images)} графиков")
    print(f"Изображения сохранены в plot_images.pkl")
    print(f"Ключи в словаре: {list(images.keys())}")
    
    return images


if __name__ == '__main__':
    images = generate_all_plots()
    print("\nГенерация графиков завершена!")
