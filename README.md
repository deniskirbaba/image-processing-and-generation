# HW 1 - Cityscapes Segmentation

## Task

Решаю задачу semantic segmentation. Цель - самостоятельно имплементировать архитектуру U-Net (конечно с меньшим количеством параметров,mac чтобы обучить это средствами Google Colab за адекватное время).

В качестве датасета - возьму Cityscapes. Описание:
> Cityscapes is a large-scale database which focuses on semantic understanding of urban street scenes. It provides semantic, instance-wise, and dense pixel annotations for 30 classes grouped into 8 categories (flat surfaces, humans, vehicles, constructions, objects, nature, sky, and void). The dataset consists of around 5000 fine annotated images and 20000 coarse annotated ones. Data was captured in 50 cities during several months, daytimes, and good weather conditions. It was originally recorded as video so the frames were manually selected to have the following features: large number of dynamic objects, varying scene layout, and varying background.

Датасет буду использовать не весь, а только некоторую часть.

## Dataset

[Cityscapes](https://www.cityscapes-dataset.com/)

Код для работы с датасетом (загрузка, визуализация данных): [data_handler.py](/src/cityscapes_data_handler.py).

Краткий анализ датасета был проведен в [eda.ipynb](/eda.ipynb).

Данные:

* train: 2915 images + labels
* val: 250 images + labels
* test: 250 images + labels

(Note: Because getting scores on the test split requires loading predictions on <https://www.cityscapes-dataset.com/> cite (and it has a large cooldown of 48h for 1 submit). So I will use part of the val dataset as the test set.)

Число классов было выбрано = 19 (виды классов есть [тут](/src/cityscapes_data_handler.py))

## Metrics calculation

Расчет метрик реализован в файле [metrics.py](/src/metrics.py).

Метрики:

* Precision - как по всем классам, так и усредненный по типу weighted, macro
* Recall - как по всем классам, так и усредненный по типу weighted, macro
* F1 score - как по всем классам, так и усредненный по типу macro
* Dice score - как по всем классам, так и усредненный по типу samples

Для реализации метрик использовалась библиотека `pytorch-ignite`.

## Logging

Логгирование выполнялось с помощью `wandb`.

## Train, val, test pipeline

Пайплайн обучения, валидации и тестирования находится в [model_pipeline.py](/src/trainer.py)

## U-Net

Реализовал классическую архитектуру U-Net ([ориг статья](https://arxiv.org/abs/1505.04597)). Реализация в [unet.py](/src/unet.py).

![unet](/media/u-net-architecture.png)

> Note: везде использую тип данных `bfloat16`.

Основной ноутбук с кодом: [unet.ipynb](/unet.ipynb).

Всего выполнил 3 эксперимента, в которых постепенно изменял архитектуру модели и конфигурацию тренировки. Все эксперименты описаны в соответствующих разделах в [unet.ipynb](/unet.ipynb).

#### Experiments

|              | img_size   |   start_channels |   depth |   epochs |   batch_size |   learning_rate |
|:-------------|:-----------|-----------------:|--------:|---------:|-------------:|----------------:|
| experiment 1 | (64, 128)  |                8 |       2 |        5 |          256 |           0.001 |
| experiment 2 | (64, 128)  |               16 |       3 |        7 |          256 |           0.001 |
| experiment 3 | (128, 256) |               16 |       4 |       10 |          128 |           0.001 |

### Experiment 1

#### Цель

Обучить модель с архитектурой U-Net и следующими параметрами:

* start_channels=8
* depth=2

Также будем использовать размер изобаржения = (64, 128).

Параметры обучения:

* Adam с learning_rate=1e-3
* epochs=5
* batch_size=256

#### Идея

Обучить достаточно малую модель на задачу semantic segmentation на датасете Cityscapes. Хочется увидеть рост метрик на основных классам сегментации (road, car, sidewalk, building, sky, terrain, person). Понятное дело, что на классах, которые встречаются мало и в целом занимают малую площадь на картинках (из-за того что я уменьшил размер картинки их плозадь еще меньше) метрики будут околонулевые.

#### Результаты

##### Графики лосс-функции при обучении (на тренировочном и валидационном датасете)

![exp1_losses](/media/exp1/W&B%20Chart%201_9_2025,%2012_15_27%20AM.png)

##### Аггрегированные (по всем классам) метрики на валидационной выборке

![exp1_val_metrics](/media/exp1/W&B%20Chart%201_9_2025,%2012_17_09%20AM.png)

##### Метрики отдельных на валидационной выборке

Классы с наиболее высокими метриками

![exp_1_val_metrics_road](/media/exp1/W&B%20Chart%201_9_2025,%2012_31_26%20AM.png)

![exp_1_val_metrics_building](/media/exp1/W&B%20Chart%201_9_2025,%2012_31_13%20AM.png)

![exp_1_val_metrics_sky](/media/exp1/W&B%20Chart%201_9_2025,%2012_31_26%20AM%20(2).png)

![exp_1_val_metrics_vegetation](/media/exp1/W&B%20Chart%201_9_2025,%2012_31_26%20AM%20(3).png)

У остальных классов метрики либо нулевые, либо очень малы

![exp_1_val_metrics_f1_other](/media/exp1/W&B%20Chart%201_9_2025,%2012_31_26%20AM%20(4).png)

##### Метрики на тестовом датасете

|               |   precision |   recall |       f1 |     dice |
|:--------------|------------:|---------:|---------:|---------:|
| aggregated    |    0.194871 | 0.203552 | 0.180893 | 0.180893 |
| road          |    0.774    | 0.995    | 0.871    | 0.871    |
| sky           |    0.767    | 0.888    | 0.823    | 0.823    |
| vegetation    |    0.836    | 0.808    | 0.822    | 0.822    |
| building      |    0.775    | 0.851    | 0.811    | 0.811    |
| sidewalk      |    0.22     | 0.031    | 0.055    | 0.055    |
| truck         |    0.018    | 0.283    | 0.033    | 0.033    |
| car           |    0.306    | 0.011    | 0.02     | 0.02     |
| fence         |    0.005    | 0.001    | 0.002    | 0.002    |
| pole          |    0.002    | 0        | 0        | 0        |
| traffic_light |    0        | 0        | 0        | 0        |
| wall          |    0        | 0        | 0        | 0        |
| terrain       |    0        | 0        | 0        | 0        |
| traffic_sign  |    0        | 0        | 0        | 0        |
| rider         |    0        | 0        | 0        | 0        |
| person        |    0        | 0        | 0        | 0        |
| bus           |    0        | 0        | 0        | 0        |
| train         |    0        | 0        | 0        | 0        |
| motorcycle    |    0        | 0        | 0        | 0        |
| unlabelled    |    0        | 0        | 0        | 0        |

##### Примеры сегментации

![exp1_1](media/exp1/image.png)
![exp1_2](media/exp1/image-1.png)
![exp1_3](media/exp1/image-2.png)
![exp1_4](media/exp1/image-3.png)
![exp1_5](media/exp1/image-4.png)

#### Выводы

В данном эксперименте была взята малая модель (120k параметров), однако результаты показали, что архитектура UNet была реализована верно и метрики растут на классах, которые занимают большую площадь в среднем на обучающих изображениях.

А именно, метрики recall и precision больше `0.8` на таких классах как

* road
* sky
* vegetation
* building

Эти классы, очевидно, наиболее представлены в датасете, поэтому модель наиболее их и выучила.

Также стоит помнить, что в целях ускорения обучения, а также из-за отсутствия больших вычислительных ресурсов изображения были уменьшены с размера 2048х1024 до 64х128. И из-за этого находить объекты, которые занимали малую площадь на изображениях исходного размера стало ещё труднее на уменьшенных изображениях.

### Experiment 2

#### Цель

Обучить модель с той же архитектурой U-Net, но с большим числом параметров и большим числом тренировочных эпох.

Изменения в сравнении с экспериментом №1:

* start_channels: 8 -> 16
* depth: 2 -> 3
* epochs: 5 -> 7

Размер изображения остался тот же: (64, 128).

#### Идея

В предыдущем эксперименте было видно, что из-за малого числа параметров модель не могла эффективно генерализироваться на обучающую выборку. Поэтому имеется следующая идея: увеличить размер модели и ожидать улучшения метрик на основных классах (road, car, sidewalk, building, sky, terrain, person).

#### Результаты

##### Графики лосс-функции при обучении (на тренировочном и валидационном датасете)

![exp2_losses](/media/exp2/W&B%20Chart%201_9_2025,%204_22_07%20PM.png)

##### Аггрегированные (по всем классам) метрики на валидационной выборке

![exp2_val_metrics](/media/exp2/W&B%20Chart%201_9_2025,%204_24_15%20PM.png)

##### Метрики отдельных на валидационной выборке

Классы с наиболее высокими метриками

![exp_2_val_metrics_road](/media/exp2/W&B%20Chart%201_9_2025,%204_26_53%20PM.png)

![exp_2_val_metrics_building](/media/exp2/W&B%20Chart%201_9_2025,%204_26_24%20PM.png)

![exp_2_val_metrics_sky](/media/exp2/W&B%20Chart%201_9_2025,%204_27_29%20PM.png)

![exp_2_val_metrics_vegetation](/media/exp2/W&B%20Chart%201_9_2025,%204_27_55%20PM.png)

![exp_2_val_metrics_car](/media/exp2/W&B%20Chart%201_9_2025,%204_27_55%20PM%20(2).png)

У остальных классов метрики либо нулевые, либо очень малы

![exp_2_val_metrics_f1_other](/media/exp2/W&B%20Chart%201_9_2025,%204_33_45%20PM.png)

##### Метрики на тестовом датасете

|               |   precision |   recall |       f1 |     dice |
|:--------------|------------:|---------:|---------:|---------:|
| aggregated    |    0.228789 | 0.222056 | 0.205045 | 0.205045 |
| road          |    0.924    | 0.932    | 0.928    | 0.928    |
| sky           |    0.788    | 0.803    | 0.795    | 0.795    |
| building      |    0.592    | 0.926    | 0.722    | 0.722    |
| vegetation    |    0.95     | 0.475    | 0.633    | 0.633    |
| car           |    0.402    | 0.893    | 0.554    | 0.554    |
| sidewalk      |    0.424    | 0.19     | 0.262    | 0.262    |
| wall          |    0.2      | 0        | 0        | 0        |
| traffic_light |    0        | 0        | 0        | 0        |
| pole          |    0        | 0        | 0        | 0        |
| traffic_sign  |    0        | 0        | 0        | 0        |
| fence         |    0        | 0        | 0        | 0        |
| terrain       |    0        | 0        | 0        | 0        |
| person        |    0.067    | 0        | 0        | 0        |
| rider         |    0        | 0        | 0        | 0        |
| truck         |    0        | 0        | 0        | 0        |
| bus           |    0        | 0        | 0        | 0        |
| train         |    0        | 0        | 0        | 0        |
| motorcycle    |    0        | 0        | 0        | 0        |
| unlabelled    |    0        | 0        | 0        | 0        |

##### Примеры сегментации

![exp2_1](media/exp2/0.png)
![exp2_2](media/exp2/1.png)
![exp2_3](media/exp2/2.png)
![exp2_4](media/exp2/3.png)
![exp2_5](media/exp2/4.png)

#### Выводы

В данном эксперименты было увеличено количество параметров в модели с 120k (experiment 1) до ~2kk. Эффективность модели выросла: значения аггрегированных метрик на тестовом датасете увеличились каждая на 3%.

Что более важно, это то, что увеличилось число классов, метрики которых ненулевые. Изменения по сравнению с экспериментом 1 следующие (рассматриваю F1 метрику):

* road: +0.05
* sky: -0.02
* vegetation: -0.2
* building: -0.08
* car: +0.53
* sidewalk: +0.2

Эти классы наиболее представлены в датасете, поэтому модель смогла их выучить.

Дальнейшие улучшения следует ожидать при увеличении параметров модели и одновременно увеличении размера изображений, чтобы дать возможность модели находить объекты классов, занимающих маленькую площадь -> значения аггрегированных метрик будут увеличиваться.

### Experiment 3

#### Цель

#### Идея

#### Результаты

#### Выводы

## Useful

<https://arxiv.org/abs/1505.04597>

<https://arxiv.org/abs/2302.06378>

<https://arxiv.org/pdf/2305.03273>

<https://paperswithcode.com/method/vision-transformer>

<https://habr.com/ru/articles/599057/>
