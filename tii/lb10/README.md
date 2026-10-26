# Работа 10

Выполненный ноутбук — 10_student.ipynb, отчёт — З10_Отчет.docx.
Сдать также 10_validation_threshold_grid.csv и 10_final_test_once.csv.
Группу и фамилию добавьте в имя DOCX перед сдачей.

Для проверки результатов без нового обращения к модели на test из корня проекта:

```powershell
& .\.venv\Scripts\python.exe tii/lb10/verify_results.py
```

Для технического воспроизведения всего ноутбука:

```powershell
& .\.venv\Scripts\python.exe tii/lb10/run_notebook.py
```

Такой повтор не создаёт новую независимую оценку: после первой проверки test нельзя менять решения по его результатам. В каждом проходе ноутбука ровно один predict_proba(Xtest), затем две метрики для заранее заданных порогов.

Окружение Python 3.12 находится в корневой .venv. Для восстановления: `py -3.12 -m venv .venv`, затем `& .\.venv\Scripts\python.exe -m pip install -r tii/lb10/requirements.txt`.
CSV, словарь, разбиение и reference validation-прогнозы скопированы из lb9. Модуль признаков совпадает с lb9. При ручном запуске рабочая папка — tii/lb10.

Выбранный на validation порог 0,15 хранится в 10_frozen_decision.json вместе с протоколом. При загрузке доверенного 10_final_model.joblib необходим tyumgu_feature_rules.py. Используйте predict_proba и сравнение с выбранным порогом: стандартный predict не применяет бизнес-порог автоматически.
Проверка всех требований — 10_checklist.md. Рублёвые стоимости являются учебными.
