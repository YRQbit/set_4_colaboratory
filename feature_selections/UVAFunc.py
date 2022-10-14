import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def UVA_numeric(data,distplot=True,dist_bins=10,lineplot=False,scatterplot=False):
  """
  # Источник
  # https://www.helenkapatsa.ru/normalnoie-raspriedielieniie/
  """

  var_group = data.columns # Список столбцов
  size = len(var_group) # Количество столбцов (3)
  plt.figure(figsize = (9 * size, 9), dpi = 300) # Параметры графика

  # Применяем расчеты к каждому столбцу
  for j,i in enumerate(var_group):
        
        # Рассчитываем основные статистические метрики
        mini = data[i].min()
        maxi = data[i].max()
        ran = data[i].max()-data[i].min() # Диапазон значений
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std() # Стандартное отклонение
        skew = data[i].skew() # Скошенность 
        kurt = data[i].kurtosis() # Эксцесс

        # Расчет точек стандартного отклонения
        points = mean - st_dev, mean + st_dev

        # Построим график с каждым из трех наборов даннных
        #Plotting the variable with every information
        plt.subplot(1, size, j+1);
        sns.distplot(data[i], 
                     hist = True, 
                     kde=  True,
                     bins=dist_bins);
        
        if lineplot == True:
          sns.lineplot(points, [1,0], color = 'black', label = "std_dev", ci=None);
        if scatterplot == True:
          sns.scatterplot([mini,maxi], [1,0], color = 'orange', label = "min/max", ci=None);
          sns.scatterplot([mean], [1], color = 'red', label = "mean", ci=None);
          sns.scatterplot([median], [1], color = 'blue', label = "median", ci=None);
        
        plt.xlabel('{}'.format(i), fontsize = 20);
        plt.ylabel('density');
        plt.title('Стандартное отклонение = {}; \
        Эксцесс = {};\n \
        Скошенность = {}; \
        Разброс, шаг гистограммы = {}\n \
        Среднее = {}; \
        Медиана = {}'.format((round(points[0],2),round(points[1],2)),
                             round(kurt,2),round(skew,2),
                             (round(mini,2),round(maxi,2),round(ran,2)),
                             round(mean,2),round(median,2)));
