import os
import sys

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Импортируем и запускаем основной скрипт
import generate_plots

if __name__ == '__main__':
    generate_plots.plot_task1()
    print("Задание 1 готово")
    generate_plots.plot_task2()
    print("Задание 2 готово")
    generate_plots.plot_task3()
    print("Задание 3 готово")
    generate_plots.plot_task4()
    print("Задание 4 готово")
    generate_plots.plot_task7()
    print("Задание 7 готово")
    print("Все графики сгенерированы!")


