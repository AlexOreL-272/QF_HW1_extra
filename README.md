# ДЗ 1* Математические модели и методы в финансах

## Описание:
Реализован калькулятор для расчета дельты опциона. На текущий момент в ветке `main` лежит код, реализующий минимальные требования (то есть расчет дельты европейского опциона и геометрического Броуновского процесса).

В дальнейшем код может быть расширен добавлением поддержки других видов опционов и процессов.

## Архитектура:
- Класс `Underlying`, представляющий собой некоторый дериватив.

- Абстрактный класс `BaseOption`, задающий базовый интерфейс для опционов.

- Класс `EuropeanOption`, реализующий `BaseOption`, представляет европейский опцион.

- Класс `BaseProcess`, задающий базовый интерфейс для процессов.

- Класс `BrownianMotionProcess`, реализующий `BaseProcess`, представляет геометрический Броуновский процесс для моделирования поведения дериватива.

- Класс `Calculator`, реализующий калькулятор для расчета дельты опциона.

## Использование (Ubuntu):
Для запуска программы необходимо выполнить следующие действия:

0. Убедиться, что установлен `CMake`, `Make` и `libtorch`.
1. Склонировать репозиторий.
2. Перейти в директорию, куда склонирован репозиторий.
3. Выполнить команду `mkdir build && cd build && cmake .. && make`.
4. Выполнить команду `./QF_HW1_extra`.
5. Программа выведет результат расчета дельты опциона для некоторых заранее прописанных параметров.

## Планы на развитие:
- Реализовать UI для интерактивного ввода параметров дериватива, процесса и опциона.

- Реализовать поддержку других видов опционов и процессов (классы-наследники от `BaseOption` и `BaseProcess` соответственно).