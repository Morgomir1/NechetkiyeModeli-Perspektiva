import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Настройка для цветного вывода графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Генерация синтетических данных временного ряда
# Создаем временной ряд с трендом и случайной составляющей
np.random.seed(42)
n = 365  # Год данных
dates = pd.date_range(start='2011-01-01', periods=n, freq='D')

# Создаем временной ряд с трендом, сезонностью и случайной составляющей
trend = np.linspace(1000, 2000, n)
seasonal = 200 * np.sin(2 * np.pi * np.arange(n) / 365.25)
noise = np.random.normal(0, 100, n)
tsday = trend + seasonal + noise

# Сохраняем данные в DataFrame
tsday_series = pd.Series(tsday, index=dates)

# 1. График исходного временного ряда
def plot_original_ts():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tsday_series.index, tsday_series.values, linewidth=1, color='blue')
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Значение', fontsize=12)
    ax.set_title('Исходный временной ряд', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tsday_original.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График исходного временного ряда сохранен")

# 2. Первые разности
def calculate_differences():
    tsday_diff = tsday_series.diff().dropna()
    
    # График первых разностей
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tsday_diff.index, tsday_diff.values, linewidth=1, color='green')
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Первая разность', fontsize=12)
    ax.set_title('Временной ряд первых разностей', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tsday_diff.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Тест Дики-Фуллера для первых разностей
    adf_result = adfuller(tsday_diff.dropna())
    
    print("\nРезультаты теста Дики-Фуллера для первых разностей:")
    print(f"Dickey-Fuller статистика: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"Критические значения:")
    for key, value in adf_result[4].items():
        print(f"  {key}: {value:.4f}")
    
    return tsday_diff, adf_result

# 3. ACF и PACF для первых разностей
def plot_acf_pacf(tsday_diff):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ACF
    plot_acf(tsday_diff.dropna(), ax=ax1, lags=40, color='blue', alpha=0.7)
    ax1.set_title('Автокорреляционная функция (ACF)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # PACF
    plot_pacf(tsday_diff.dropna(), ax=ax2, lags=40, color='red', alpha=0.7)
    ax2.set_title('Частная автокорреляционная функция (PACF)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acf_pacf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Графики ACF и PACF сохранены")

# 4. Спектрограмма
def plot_spectrogram(tsday_diff):
    # Вычисляем периодограмму
    frequencies, spectrum = signal.periodogram(tsday_diff.dropna(), fs=1.0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(frequencies, spectrum, color='purple', linewidth=1)
    ax.set_xlabel('Частота', fontsize=12)
    ax.set_ylabel('Спектральная плотность', fontsize=12)
    ax.set_title('Спектрограмма временного ряда первых разностей', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('spectrogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Спектрограмма сохранена")
    
    return frequencies, spectrum

# 5. Подбор модели ARIMA
def fit_arima_model():
    # Автоматический подбор модели ARIMA
    best_aic = np.inf
    best_model = None
    best_order = None
    
    # Перебираем различные комбинации параметров
    for p in range(0, 6):
        for d in range(0, 4):
            for q in range(0, 6):
                try:
                    model = ARIMA(tsday_series, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        best_order = (p, d, q)
                except:
                    continue
    
    print(f"\nЛучшая модель ARIMA: {best_order}")
    print(f"AIC: {best_aic:.2f}")
    print(f"\nПараметры модели:")
    print(best_model.summary())
    
    return best_model, best_order

# 6. Оценка точности модели
def evaluate_model(model):
    fitted_values = model.fittedvalues
    residuals = model.resid
    
    # Вычисляем метрики
    n = len(residuals)
    me = np.mean(residuals)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    
    # MPE и MAPE
    actual = tsday_series[fitted_values.index]
    mpe = np.mean((residuals / actual) * 100)
    mape = np.mean(np.abs(residuals / actual) * 100)
    
    # MASE (Mean Absolute Scaled Error)
    naive_forecast_error = np.abs(tsday_series.diff().dropna())
    mase = mae / np.mean(naive_forecast_error)
    
    # ACF1 остатков
    acf1 = acf(residuals.dropna(), nlags=1)[1]
    
    metrics = {
        'ME': me,
        'MAE': mae,
        'RMSE': rmse,
        'MPE': mpe,
        'MAPE': mape,
        'MASE': mase,
        'ACF1': acf1
    }
    
    print("\nМетрики точности модели:")
    print(f"ME:  {me:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MPE: {mpe:.4f}%")
    print(f"MAPE: {mape:.4f}%")
    print(f"MASE: {mase:.4f}")
    print(f"ACF1: {acf1:.4f}")
    
    return metrics

# 7. Прогноз на 30 дней вперед
def forecast_30_days(model):
    forecast_steps = 30
    forecast = model.forecast(steps=forecast_steps)
    forecast_ci = model.get_forecast(steps=forecast_steps).conf_int()
    
    # Создаем даты для прогноза
    last_date = tsday_series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    
    # График с прогнозом
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Исходные данные
    ax.plot(tsday_series.index, tsday_series.values, linewidth=1.5, color='blue', label='Исходные данные')
    
    # Прогноз
    ax.plot(forecast_dates, forecast, linewidth=2, color='red', label='Прогноз', linestyle='--')
    
    # Доверительные интервалы
    ax.fill_between(forecast_dates, 
                     forecast_ci.iloc[:, 0], 
                     forecast_ci.iloc[:, 1], 
                     color='red', alpha=0.2, label='95% доверительный интервал')
    
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Значение', fontsize=12)
    ax.set_title('Прогноз на 30 дней вперед', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast_30_days.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График прогноза на 30 дней сохранен")
    
    return forecast, forecast_ci, forecast_dates

# Основная функция
def main():
    print("Начало анализа временных рядов...")
    
    # 1. График исходного ряда
    plot_original_ts()
    
    # 2. Первые разности и тест Дики-Фуллера
    tsday_diff, adf_result = calculate_differences()
    
    # 3. ACF и PACF
    plot_acf_pacf(tsday_diff)
    
    # 4. Спектрограмма
    frequencies, spectrum = plot_spectrogram(tsday_diff)
    
    # 5. Подбор модели ARIMA
    model, order = fit_arima_model()
    
    # 6. Оценка точности
    metrics = evaluate_model(model)
    
    # 7. Прогноз на 30 дней
    forecast, forecast_ci, forecast_dates = forecast_30_days(model)
    
    print("\nАнализ завершен!")
    
    return {
        'tsday_series': tsday_series,
        'tsday_diff': tsday_diff,
        'adf_result': adf_result,
        'model': model,
        'order': order,
        'metrics': metrics,
        'forecast': forecast,
        'forecast_ci': forecast_ci,
        'forecast_dates': forecast_dates
    }

if __name__ == '__main__':
    results = main()

