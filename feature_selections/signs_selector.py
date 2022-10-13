import pandas as pd

def feature_selector(df_features,df_target=pd.DataFrame(), std_upper=0.000001, skew_upper=0.000001, corr_min=0.01, corr_max=1.0, returned="result_table"):
  """
  df_features            :: фрейм исследуемых переменных (выборка)
  df_target=pd.DataFrame() :: предполагает фрейм из одной (1) переменной (столбец)
  std_upper=0.000001     :: верхний предел допуска примет значение по умолчанию == среднее по всем показателям
  skew_upper=0.000001    :: верхний предел допуска примет значение по умолчанию == среднее по всем показателям
  corr_min=0.01          :: по умолчанию нижний предел допуска 0.01
  corr_max=1.0           :: по умолчанию верхний предел допуска 1.0
  returned="result_table" :: summary_table \ result_table \ comliance_table
  
  df_comliance = df_results[ 
      ( df_results["std_value"] < std_upper )&
      ( df_results["skew_value"] < skew_upper )&
      ( df_results["corr_value"] > corr_min )&
      ( df_results["corr_value"] < corr_max )] 

  """

  df_std = pd.DataFrame()
  std_value = {}
  skew_value = {}

  # Наполнение словарей
  # 
  count_features=0
  for el in df_features.columns:
    # .
    std_value[el] = df_features[el].std().round(3)
      # Добавление пары ключ:значение в dict-словарь
      # .
    skew_value[el] = df_features[el].skew().round(3)
      # Добавление пары ключ:значение в dict-словарь
      # .

    count_features+=1
  
  # Добавление инфо-строк в df_features
  # 
  df_std = df_std.append(pd.Series(data = std_value), ignore_index=True)  
  df_std = df_std.append(pd.Series(data = skew_value), ignore_index=True)
  
  # Транспонирование df_features
  # 
  df_std = df_std .T.rename(columns = {0:"std_value",1:"skew_value"})
    # Транспонирование фрейма + переименование столбцов(переменных)

  # Добавление пустой строки в df_features
  # 
  df_std = df_std.append(pd.Series(name=" "), ignore_index=False)
  df_std = df_std.fillna("")

  # Вычисление средних по std_value и skew_value
  # 
  exclude_idx = df_std.index.isin([" "])
    # получение списка индексов
    # .
  mean_std = df_std[~exclude_idx].mean().values[0]
    # df_std[~exclude_idx] ==> исключение строк df_features по индексу
    # .
  mean_skew = df_std[~exclude_idx].mean().values[1]
    # df_std[~exclude_idx] ==> исключение строк df_features по индексу
    # .
  means_val = {"std_value":mean_std,
              "skew_value":mean_skew}
              # Формирование словаря
  
  # Добавление инфо-строки в df_features
  # 
  df_std = df_std.append(pd.Series(data = means_val,name="mean_value"), ignore_index=False)

  
  # Вычисление показателей корреляции
  # 
  if (len(df_target.columns) != 0) and (len(df_target.index) != 0):
    # В случае, когда Таргет указан явно

    target_name = df_target.columns[0]

    # Формирование фрейма корреляции
    # 
    df_correlation = df_features.join(df_target, how="left", lsuffix=target_name,rsuffix='')
      # Добавление столбца из dframe в df через join
      # .
    df_correlation = df_correlation.corr()
      # Создание талицы корреляции переменных
      # .

    df_results = df_std.iloc[:count_features]
      # Достаем переменные из фрейма сводного отчета
      # .
    df_results["corr_value"] = df_correlation.iloc[:count_features+1][target_name]
      # Добавляем переменную с показателями корреляции
      # .


    if (std_upper == 0.000001): std_upper = mean_std
    if (skew_upper == 0.000001): skew_upper = mean_skew

    # Формирование выборки по условиям
    # 
    df_comliance = df_results[ ( df_results["std_value"] < std_upper )&( df_results["skew_value"] < mean_skew ) &( df_results["corr_value"] > corr_min )&( df_results["corr_value"] < corr_max )] 

    df_results = df_results.fillna("")
      # Преобразовываем NaN-значения в пустые значения
    
    if returned == "summary_table": 
      return df_std
    elif returned == "result_table": 
      return df_results
    elif returned == "comliance_table":
      return df_comliance


  else:
    # В случае, когда Таргет не указан явно, используем 
    # как Таргет последний столбец
    # .
    df_correlation = df_features.corr()
      # Формируем фрейм показателей корреляции
      # .
    target_name = df_correlation.columns[-1]
      # В качестве Таргета принимаем последний столбец
      # .
    df_results = df_std.iloc[:count_features]
      # Достаем переменные из фрейма сводного отчета
      # .
    df_results["corr_value"] = df_correlation.iloc[:count_features][target_name]
      # Добавляем переменную с показателями корреляции
      # .

    if (std_upper == 0.000001): std_upper = mean_std
    if (skew_upper == 0.000001): skew_upper = mean_skew

    # Формирование выборки по условиям
    # 
    df_comliance = df_results[ ( df_results["std_value"] < std_upper )&( df_results["skew_value"] < mean_skew ) &( df_results["corr_value"] > corr_min )&( df_results["corr_value"] < corr_max )] 
    
    if returned == "summary_table": 
      return df_std
    elif returned == "result_table": 
      return df_results
    elif returned == "comliance_table":
      return df_comliance


def get_cube_root(x):
  """
  # Источник:
  # https://pythonru.com/osnovy/kak-izvlech-kubicheskij-koren-v-python
  """
    if x < 0:
        x = abs(x)
        cube_root = x**(1/3)*(-1)
    else:
        cube_root = x**(1/3)
    return cube_root    


def pvalue_selector(dframe):
  """
  """
  from scipy import stats

  df_results = pd.DataFrame()
  df_summary = pd.DataFrame()
  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  base
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  for column in dframe.columns:
    
      pvalue_jarque_bera.append(stats.jarque_bera(dframe[column]).pvalue)

      stat, p_val = stats.normaltest(dframe[column])
      pvalue_pearson.append(p_val)

      v_ariance = round(dframe[column].var(),3)
      variance_lst.append(v_ariance)
  
  df_summary[f"pvalue_jarque_bera"] = pd.Series(pvalue_jarque_bera, index=[f"{el}" for el in dframe.columns])
  df_summary[f"pvalue_pearson"] = pd.Series(pvalue_pearson, index=[f"{el}" for el in dframe.columns])   
  df_summary[f"variance"] = pd.Series(variance_lst, index=[f"{el}" for el in dframe.columns])


  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  math.sqrt(x)
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  import math

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]

  for column in dframe.columns:
    try:
      
      # Выполним извлечение квадратного корня:
      # .
      result = [math.sqrt(el) for el in dframe[column]]
      ser_sqrt = pd.Series(result)
      
      data_norm = pd.DataFrame({column:result}, index = dframe.index)
      
      pvalue_jarque_bera.append(stats.jarque_bera(ser_sqrt).pvalue)

      stat, p_val = stats.normaltest(ser_sqrt)
      pvalue_pearson.append(p_val)

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  # cvb = pd.DataFrame({"pvalue_jarque_bera": pd.Series(pvalue_jarque_bera, index=[f"{el}_sqrt" for el in dframe.columns]),
  #                     "pvalue_pearson": pd.Series(pvalue_pearson, index=[f"{el}_sqrt" for el in dframe.columns])
  #                     })

  
  df_summary[f"variance"] = pd.Series(v_ariance, index=[f"{el}" for el in dframe.columns])
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      }, index=[f"{el}_sqrt" for el in dframe.columns])

  df_summary = df_summary.append(df_res)


  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  x**(1/3)
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  for column in dframe.columns:
    try:
      
      # Выполним извлечение кубического корня:
      # .
      result = [get_cube_root(el) for el in dframe[column]]
      
      data_norm = pd.DataFrame({column:result}, index = dframe.index)
      
      pvalue_jarque_bera.append(round(stats.jarque_bera(result).pvalue,5))

      stat, p_val = stats.normaltest(result)
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  # cvb = pd.DataFrame({"pvalue_jarque_bera": pd.Series(pvalue_jarque_bera, index=[f"{el}_sqrt" for el in dframe.columns]),
  #                     "pvalue_pearson": pd.Series(pvalue_pearson, index=[f"{el}_sqrt" for el in dframe.columns])
  #                     })
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_cbrt" for el in dframe.columns])

  df_summary = df_summary.append(df_res)

  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  log(x)
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  for column in dframe.columns:
    try:
      # Выполним извлечение кубического корня:
      # .
      result = np.log(dframe[column])

      data_norm = pd.DataFrame({column:result}, index = dframe.index)
      
      pvalue_jarque_bera.append(round(stats.jarque_bera(result).pvalue,5))

      stat, p_val = stats.normaltest(result)
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_log" for el in dframe.columns])

  df_summary = df_summary.append(df_res)



  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  MinMax_scaler
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """
  
  from sklearn.preprocessing import MinMaxScaler

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  for column in dframe.columns:
    try:
      # Выполним извлечение кубического корня:
      # .
      
      data_norm_fit = MinMaxScaler().fit(dframe[column].values.reshape(-1,1))
      data_norm = pd.DataFrame(data_norm_fit.transform(dframe[column].values.reshape(-1,1)), 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)

      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  

  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_min_max" for el in dframe.columns])

  df_summary = df_summary.append(df_res)

  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  Standart_scaler
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  from sklearn.preprocessing import StandardScaler

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  for column in dframe.columns:
    try:
      
      data_norm_fit = StandardScaler().fit(dframe[column].values.reshape(-1,1))
      data_norm = pd.DataFrame(data_norm_fit.transform(dframe[column].values.reshape(-1,1)), 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_standart" for el in dframe.columns])

  df_summary = df_summary.append(df_res)

  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  MaxAbs_scaler
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  from sklearn.preprocessing import MaxAbsScaler

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  for column in dframe.columns:
    try:
      
      data_norm_fit = MaxAbsScaler().fit(dframe[column].values.reshape(-1,1))
      data_norm = pd.DataFrame(data_norm_fit.transform(dframe[column].values.reshape(-1,1)), 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_max_abs" for el in dframe.columns])

  df_summary = df_summary.append(df_res)

  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  Robust_scaler
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  from sklearn.preprocessing import RobustScaler

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  for column in dframe.columns:
    try:
      
      data_norm_fit = RobustScaler().fit(dframe[column].values.reshape(-1,1))
      data_norm = pd.DataFrame(data_norm_fit.transform(dframe[column].values.reshape(-1,1)), 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))
      
      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)

    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_robust" for el in dframe.columns])

  df_summary = df_summary.append(df_res)

  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  QuantileTransformer
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  from sklearn.preprocessing import QuantileTransformer

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  for column in dframe.columns:
    try:
      qqTransform = QuantileTransformer(output_distribution="normal", random_state = 0)

      data_norm_fit_transform = qqTransform.fit_transform(dframe[column].values.reshape(-1,1))
      
      
      data_norm = pd.DataFrame(data_norm_fit_transform, 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_qqTransform" for el in dframe.columns])

  df_summary = df_summary.append(df_res)

  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  PowerTransformer
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  from sklearn.preprocessing import PowerTransformer

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  
  for column in dframe.columns:
    try:
      
      if min(dframe[column].values) <= 0 :
        pwrTransform = PowerTransformer(method="yeo-johnson", standardize = False)
        pwrTransform_name = "pwrTransform_y"
      else:
        pwrTransform = PowerTransformer(method="box-cox", standardize = False)
        pwrTransform_name = "pwrTransform_b"

      data_norm_fit_transform = pwrTransform.fit_transform(dframe[column].values.reshape(-1,1))
      
      data_norm = pd.DataFrame(data_norm_fit_transform, 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_{pwrTransform_name}" for el in dframe.columns])

  df_summary = df_summary.append(df_res)

  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  Normalizer_column
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  from sklearn.preprocessing import normalize

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]

  for column in dframe.columns:
    try:
      
      data_norm_fit = normalize([dframe[column].values])

      data_norm = pd.DataFrame(data_norm_fit[0], 
                               index = dframe[column].index,
                               columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(data_norm[column].var(),3)
      variance_lst.append(v_ariance)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst
                      }, index=[f"{el}_normalize" for el in dframe.columns])

  df_summary = df_summary.append(df_res)


  return df_summary

# pvalue_selector(df_pandas)
