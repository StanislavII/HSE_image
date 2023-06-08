# HSE image project

## Ilyushin Stanislav
## Dehazing images using DCP

### Введение. Состав проекта.

* _В папке images_nohaze находятся фотографии без естественного тумана на которые аугментировалась дымка для проверки качества алгоритма_
* _В папке test_images_hazed находятся фотографии с туманом_
* _Файл DCP.py реализует в себе оболочку алгоритма Dark Channel Prior_
* _Файл Guided_filter.py реализует в себе управляемый фильтер (Guided Filter + Box Filter) импортируемый в DCP.py_
* _Сам проект визуально реализован в отчете Inference.ipynb_

### Imports. Requirements

1. Библиотеки испльзующиеся для базовых алгоритмов и оболочек, а также визулизации и некоторых предобоработок фото
  * Numpy, Scipy, OpenCv, Matplotlib
2. Либа используемая для генерации тумана [imgaug](https://imgaug.readthedocs.io/en/latest/)
3. Вспомогательные либы для расчета метрик и оптимизации
  * Scikit-image, itertools
```Bash
    pip install imgaug
    pip install scikit-image
```

### Results

**Часть с естественным туманом**

Привожу выкладки из *Inference.ipynb* 


## 3 part. Airflow and postgres

Для начала нужно создать подключение к нашему хосту для мак-юзеров это делается следующим образом варианты для хоста могут быть _(localhost, postgres)_

![Connection](connection.png)

После создания подключения можно официально запускать основной даг в airflow - pre_parser.py, который имеет следующую струтуру 

![DAG](dag_screen.png)

* parser_python парсит данные едадила и сохраняет их в папку data по дням в формате csv
* create_postgres_table создает таблицу с необходимыми форматами данных
* import_file выгружает данные в дб ежедневно
* notify bash команда сообщающая об ошибке либо успехе всего процесса

Результат отработки процесса можно увидеть dbeaver

![Base](db_screen.png)

## 4 part. Container in Container with GPU Image

> NB: основной проблемой чего-то gpu-шного может быть связано с совместимостью сборок всего docker-compose, если нам нужно добавить, грубо говоря, образ tensorflow-gpu, который будет основной рабочей лошадкой всех процессов с нейронными сетками, нужно делать кастомизацию каждого образа внутри docker-compose начиная от веб-сервера, планировщика и до основного apache/airflow
