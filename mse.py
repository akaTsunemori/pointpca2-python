import numpy as np


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


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
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    4.281916073750454,
    2.0795400774637463,
    1.9625220718036183,
    2.35961623913413,
    6.073632585745445,
    4.2524905729150335,
    12.166122750442264,
    6.074337310921445,
    5.376460435151345,
    3.8060708758252093,
    1.6777516303887994,
    2.3923508320985922,
    0.802890101389266,
    0.8787449262140822,
    0.8978792577245278,
    0.6800866494534135,
    0.714543312584997,
    0.7713030023283366,
    0.9241492078217683,
    0.9308366901813235,
    0.6550992776462952,
    0.6115133070356171,
    0.5376532813369079,
    0.9210370301049452,
    0.755836905376405,
    0.6092059982261083,
    0.2814616658805408,
    0.43465280873173945
]

print(mse(y_true, y_pred))
