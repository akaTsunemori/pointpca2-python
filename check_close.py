import math


y_true = [
    0.9995,
    0.9994,
    0.9973,
    0.9490,
    0.9490,
    0.9490,
    0.0567,
    0.0567,
    0.0567,
    0.9490,
    0.9490,
    0.9490,
    4.2819,
    2.3596,
    1.9625,
    2.0795,
    6.0736,
    5.8559,
   12.1661,
    6.0743,
    6.3723,
    3.8061,
    1.6778,
    1.7021,
    0.8979,
    0.8787,
    0.8029,
    0.7713,
    0.7145,
    0.6801,
    0.9241,
    0.9308,
    0.8534,
    0.5777,
    0.7047,
    0.8095,
    0.7558,
    0.6092,
    0.1958,
    0.2541,
]

y_pred = [
1.0000,
1.0000,
1.0000,
1.0000,
1.0000,
0.9490,
0.0000,
0.0000,
0.0567,
0.9490,
1.0000,
0.9490,
4.2819,
2.3596,
1.9625,
2.0795,
6.0736,
5.8559,
12.1661,
6.0743,
6.3723,
3.8061,
1.6778,
1.7021,
0.8979,
0.8787,
0.8029,
0.7713,
0.7145,
0.6801,
0.9241,
0.9308,
0.8534,
0.5777,
0.7047,
0.8095,
0.7558,
0.6092,
0.1973,
0.6058,
]


for i in range(len(y_true)):
    if not math.isclose(y_true[i], y_pred[i], abs_tol=1e-1):
        print(y_true[i], y_pred[i])
