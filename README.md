### Dependencies
```
pip install -r requirements.txt
git clone https://github.com/ocenandor/freegroup.git
cd freegroup
git checkout general-word-problem
python setup.py install
```

### Launch

* Сначала отредактировать файл `env.json`: change **WANDB_USERNAME**, **WANDB_DIR**(optional)

* Запуск обучения (Во время первого запуска надо будет авторизоваться в wandb): `python train_whitehead.py`

Полезные инструменты есть в ноутбуках (№):
1. Генерация датасета с загрузкой и выгрузкой wandb (для запуска нужны сэмплеры, они лежат в другом месте, занимают достаточно много места)
2. Инференс модели с вспомогательными функциями
3. Если модель новая, её конфиг надо закинуть на wandb

### Optimization Methods in Machine Learning Course
Experiment template file is [007 notebook](notebooks/007_optimization_experiments.ipynb)
