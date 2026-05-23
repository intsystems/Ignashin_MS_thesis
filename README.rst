<table>
    <tr>
        <td align="center"> <b> Название исследуемой задачи </b> </td>
        <td> Frank--Wolfe algorithm in machine learning </td>
    </tr>
    <tr>
        <td align="center"> <b> Тип научной работы </b> </td>
        <td> ВКР </td>
    </tr>
    <tr>
        <td align="center"> <b> Автор </b> </td>
        <td> Игнашин Игорь Николаевич </td>
    </tr>
    <tr>
        <td align="center"> <b> Научный руководитель </b> </td>
        <td> кандидат ф.-м. наук, Грабовой Андрей Олегович </td>
    </tr>
</table>

Abstract
========

Работа посвящена развитию алгоритма Frank--Wolfe и его модификаций для задач
выпуклой оптимизации, возникающих в машинном обучении, транспортном
моделировании и распределенной оптимизации. Основное внимание уделяется
проекционно-свободным методам, которые сохраняют линейный оракул вместо
проекции, но улучшают практическую сходимость за счет памяти о направлениях,
стохастического выбора блоков и распределенного взаимодействия агентов.

Для транспортной модели Бекмана предложены N-conjugate Frank--Wolfe как
обобщение CFW/BFW на несколько прошлых направлений и Weighted Fukushima
Frank--Wolfe как модификация FFW с экспоненциальным сглаживанием. Идеи
сопряженных направлений перенесены на задачи машинного обучения без явного
вычисления гессиана: информация о кривизне аппроксимируется разностями
градиентов. Для CFW доказана оценка по минимальному зазору Frank--Wolfe,
сохраняющая порядок классического FW.

Также предложен SOFW как частный случай Block-Coordinate Frank--Wolfe для
транспортной задачи с блоками по OD-парам и доказана оценка ожидаемой
сходимости для одиночного и батчевого выбора блоков. Для распределенной
постановки предложен DBCFW, объединяющий блочный линейный оракул, консенсус и
отслеживание градиента; для него получена оценка сходимости порядка O(1/t).
Экспериментальная часть включает транспортные сети, логистическую регрессию на
Mushrooms и MNIST, а также сравнение FW-модификаций по зазору Frank--Wolfe и
времени работы.

Materials
=========

1. PDF текста работы: `paper/MasterThesis.pdf <paper/MasterThesis.pdf>`_.
2. PDF слайдов: `slides/thesis_slides.pdf <slides/thesis_slides.pdf>`_.
3. Исходник текста работы: `paper/MasterThesis.tex <paper/MasterThesis.tex>`_.
4. Исходник слайдов: `slides/thesis_slides.tex <slides/thesis_slides.tex>`_.

Software modules developed as part of the study
======================================================

1. A python code with Frank--Wolfe modifications for machine learning:
   https://github.com/ThunderstormXX/FW-in-ML
2. A python code with transport modeling and Frank--Wolfe experiments:
   https://github.com/ThunderstormXX/mmo_tm
