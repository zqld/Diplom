# NTHU-DDD Dataset Setup

## Как скачать NTHU-DDD Dataset

### Шаг 1: Регистрация

1. Перейдите на GitHub: https://github.com/CVD-YPF/NTHU-DDD-Dataset
2. Нажмите "Star" (не обязательно) 
3. Нажмите "Download" или перейдите в раздел Releases
4. Для скачивания может потребоваться авторизация через Google

### Шаг 2: Структура файлов

После скачивания у вас должен быть архив. Распакуйте его.

**Ожидаемая структура:**
```
NTHU-DDD/
└── videos/
    ├── person_01/
    │   ├── Normal_01.avi
    │   ├── Normal_02.avi
    │   ├── Sleeping_01.avi
    │   ├── Sleeping_02.avi
    │   ├── Yawning_01.avi
    │   └── Yawning_02.avi
    ├── person_02/
    │   └── ...
    └── ...
```

### Шаг 3: Куда положить

Скопируйте папку `videos/` в:
```
Diplom_Face_cloude/data/nthu_ddd/raw/
```

Итоговая структура:
```
Diplom_Face_cloude/
└── data/
    └── nthu_ddd/
        └── raw/
            └── videos/
                ├── person_01/
                │   └── ...
                └── ...
```

### Шаг 4: Запуск парсера

```bash
cd Diplom_Face_cloude
python -m neurofocus.ml.nthu_ddd_setup
```

Скрипт автоматически:
1. Найдёт видео
2. Извлечёт landmarks через MediaPipe
3. Вычислит признаки (EAR, MAR, head pose)
4. Создаст последовательности по 30 кадров
5. Объединит с синтетическими данными

### Альтернативный датасет: YawDD

Если NTHU-DDD не скачивается, можно использовать YawDD:
- URL: https://sites.google.com/site/yawddbenchmark/
- Содержит ~400 видео с зеваниями
- Проще в использовании

### Размер данных

- NTHU-DDD: ~1-2 GB
- YawDD: ~500 MB

### После скачивания

После скачивания и парсинга:

```bash
# Запустить объединение и обучение
python -c "
from neurofocus.ml.nthu_ddd_setup import merge_datasets
merge_datasets()
"

# Или запустить полный процесс
python -m neurofocus.ml.lstm_trainer
```
