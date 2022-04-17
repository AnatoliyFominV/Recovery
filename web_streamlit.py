from tensorflow import keras
from PIL import Image
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import json
import time
import tensorflow as tf
import streamlit as st

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + "/model"
model = keras.models.load_model(path)


def predict_1(dist="Приволжский", pvt="Н", carbonate=0, por=0.2, perm=0.1, visc=1):
    data = {"federal_distr": [[dist]], "pvt": [[pvt]], "carbonate_or_not": [[carbonate]], "por": [[por]],
            "perm": [[perm]], "visc_plast": [[visc]]}
    data = tf.data.Dataset.from_tensor_slices((data, [0]))
    data = data.shuffle(buffer_size=len(data))
    data.batch(1)

    return model.predict(data)[0][0]





# _______________________Web________________________________________ #
# _______________________Title_____________________________________ #
img = Image.open(dir_path + "\\rgu_icon.png")

st.image(img, width=100)
st.title("Предскажем коэффициент извлечения нефти по ГФХ")
st.write(
    """ 
    **Автор** Фомин Анатолий Витальевич   
    **Контактные данные** +7(926)4395526, kmno4.172@gmail.com   
    **Цель** Выявить зависимость КИН от ГФХ    
    **Необходимые данные** Федеральный округ, тип месторождения, пористость, проницаемость, вязкость в пластовых условиях
    ***
    """)

# _______________________Page_Layout______________________________ #
col1 = st.sidebar
col2, col3 = st.columns((10, 1))


# Sidebar
col1.header("Свойства пласта")
federal_distr = col1.selectbox("Регион:", ('Южный', 'Сибирский', 'Северо-Кавказский', 'Северо-Западный',
       'Приволжский', 'Дальневосточный', 'Уральский',
       'Шельф Российской Федерации'))
pvt = col1.selectbox("Тип месторождения:", ('Н', 'ГН', 'НГК', 'НГ'))
carbonate = col1.selectbox("Тип коллектора:", ('Терригенный', "Карбонатный"))
is_carbonate = 1 if carbonate == "Карбонатный" else 0
por = col1.slider("Пористость: ", 0.05, 0.3, 0.2, 0.001)
perm_md = col1.slider("Проницаемость мД: ", 1, 3000, 100, 1)
perm = perm_md / 1000
visc = col1.slider("Вязкость сП*с: ", 0.1, 300., 10., 0.1, "%2f")


# Предсказать КИН по данным:
pred = predict_1(dist=federal_distr, pvt=pvt, carbonate=is_carbonate, por=por, perm=perm, visc=visc)
# print("Предсказанный КИН:", pred)

st.write(
    f""" ## Предсказанный КИН на основании месторождений аналогов: {pred:.2f}
    """
)
