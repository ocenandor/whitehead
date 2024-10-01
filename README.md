Launch
```
pip install -r requirements.txt
git clone https://github.com/ocenandor/freegroup.git
cd freegroup
git checkout general-word-problem
python setup.py install
```


Сначала отредактировать файл `env.json`: change WANDB_USERNAME, WANDB_DIR(optional)

Запуск обучения (Во время первого запуска надо будет авторизоваться в wandb):
`python train_whitehead.py`


Ноутбуки:
`001` - Генерация датасета с загрузкой и выгрузкой wandb (для запуска нужны сэмплеры, они лежат в другом месте, занимают достаточно много места)
`002` - Инференс модели с вспомогательными функциями
`003` - Если модель новая, её конфиг надо закинуть на wandb

