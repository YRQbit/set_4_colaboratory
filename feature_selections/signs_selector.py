import pandas as pd
import numpy as np

def feature_selector(df_features,df_target=pd.DataFrame(), var_max="mean", mean_max="mean", skew_max="mean", corr_min=0.01, corr_max=1.0, returned="result_table"):
  """

  df_features              :: фрейм исследуемых переменных (выборка)
  df_target=pd.DataFrame() :: предполагает фрейм из одной (1) переменной (столбец)
  var_max="mean"           :: верхний предел допуска примет значение по умолчанию == среднее по всем показателям
  mean_max="mean"          :: верхний предел допуска примет значение по умолчанию == среднее по всем показателям
  skew_max=="mean"         :: верхний предел допуска примет значение по умолчанию == среднее по всем показателям
  corr_min=0.01            :: по умолчанию нижний предел допуска 0.01
  corr_max=1.0             :: по умолчанию верхний предел допуска 1.0
  returned="result_table"  :: summary_table \ result_table \ comliance_table
  
  df_comliance = df_results[ 
      ( df_results["std_value"] < std_upper )&
      ( df_results["skew_value"] < skew_upper )&
      ( df_results["corr_value"] > corr_min )&
      ( df_results["corr_value"] < corr_max )] 
  """

  df_std = pd.DataFrame()
  vari_value = {}
  std_value = {}
  mean_value={}
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
    vari_value[el] = df_features[el].var().round(3)
      # 
    mean_value[el] = df_features[el].mean().round(3)

    count_features+=1
  
  # Добавление инфо-строк в df_features
  # 
  df_std = df_std.append(pd.Series(data = vari_value), ignore_index=True)
  df_std = df_std.append(pd.Series(data = std_value), ignore_index=True)
  df_std = df_std.append(pd.Series(data = mean_value), ignore_index=True) 
  df_std = df_std.append(pd.Series(data = skew_value), ignore_index=True)
  
  # display(df_std)

  # Транспонирование df_features
  # 
  df_std = df_std .T.rename(columns = {0:"variance",
                                       1:"std_value",
                                       2:"mean_value",
                                       3:"skew_value"})
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

  

  mean_vari = df_std[~exclude_idx].mean().values[0]
    # 
  mean_std = df_std[~exclude_idx].mean().values[1]
    # df_std[~exclude_idx] ==> исключение строк df_features по индексу
    # .
  mean_mean = df_std[~exclude_idx].mean().values[2]
    # df_std[~exclude_idx] ==> исключение строк df_features по индексу
    # .
  mean_skew = df_std[~exclude_idx].mean().values[3]
    # 
  means_val = {"variance":mean_vari,
               "std_value":mean_std,
               "mean_value":mean_mean,
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


    if (var_max == "mean"): var_max = mean_vari
    if (mean_max == "mean"): mean_max = mean_mean
    if (skew_max == "mean"): skew_max = mean_skew

    # Формирование выборки по условиям
    # 
    df_comliance = df_results[ ( df_results["variance"] < var_max )&( df_results["mean_value"] < mean_max )&( df_results["skew_value"] < skew_max ) &( df_results["corr_value"] > corr_min )&( df_results["corr_value"] < corr_max )] 

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

    if (var_max == "mean"): var_max = mean_vari
    if (mean_max == "mean"): mean_max = mean_mean
    if (skew_max == "mean"): skew_max = mean_skew

    # Формирование выборки по условиям
    #
    df_comliance = df_results[ ( df_results["variance"] < var_max )&( df_results["mean_value"] < mean_max )&( df_results["skew_value"] < skew_max ) &( df_results["corr_value"] > corr_min )&( df_results["corr_value"] < corr_max )]
    
    if returned == "summary_table": 
      return df_std
    elif returned == "result_table": 
      return df_results
    elif returned == "comliance_table":
      return df_comliance

# feature_selector(df_pandas)



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

  
def transform_collection(dframe):
  """
  """
  from scipy import stats

  from statistics import variance


  df_results = pd.DataFrame()
  df_summary = pd.DataFrame()
  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  base
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  for column in dframe.columns:
    
      pvalue_jarque_bera.append(stats.jarque_bera(dframe[column]).pvalue)

      stat, p_val = stats.normaltest(dframe[column])
      pvalue_pearson.append(p_val)

      v_ariance = round(variance(dframe[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(dframe[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(dframe[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(dframe[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(dframe[column].skew(),5)
      skew_lst.append(s_kew)
  
  df_summary[f"pvalue_jarque_bera"] = pd.Series(pvalue_jarque_bera, index=[f"{el}" for el in dframe.columns])
  df_summary[f"pvalue_pearson"] = pd.Series(pvalue_pearson, index=[f"{el}" for el in dframe.columns])   
  df_summary[f"variance"] = pd.Series(variance_lst, index=[f"{el}" for el in dframe.columns])
  df_summary[f"mean"] = pd.Series(mean_lst, index=[f"{el}" for el in dframe.columns])
  df_summary[f"mode"] = pd.Series(mode_lst, index=[f"{el}" for el in dframe.columns])
  df_summary[f"median"] = pd.Series(median_lst, index=[f"{el}" for el in dframe.columns])
  df_summary[f"skew"] = pd.Series(skew_lst, index=[f"{el}" for el in dframe.columns])


  """
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  math.sqrt(x)
  ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
  """

  import math

  pvalue_jarque_bera=[]
  pvalue_pearson=[]
  variance_lst=[]
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]

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

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
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
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
  for column in dframe.columns:
    try:
      
      # Выполним извлечение кубического корня:
      # .
      result = [get_cube_root(el) for el in dframe[column]]
      
      data_norm = pd.DataFrame({column:result}, index = dframe.index)
      
      pvalue_jarque_bera.append(round(stats.jarque_bera(result).pvalue,5))

      stat, p_val = stats.normaltest(result)
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
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
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
  for column in dframe.columns:
    try:
      # Выполним извлечение кубического корня:
      # .
      result = np.log(dframe[column])

      data_norm = pd.DataFrame({column:result}, index = dframe.index)
      
      pvalue_jarque_bera.append(round(stats.jarque_bera(result).pvalue,5))

      stat, p_val = stats.normaltest(result)
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
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
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
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

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
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
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
  for column in dframe.columns:
    try:
      
      data_norm_fit = StandardScaler().fit(dframe[column].values.reshape(-1,1))
      data_norm = pd.DataFrame(data_norm_fit.transform(dframe[column].values.reshape(-1,1)), 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
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
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
  for column in dframe.columns:
    try:
      
      data_norm_fit = MaxAbsScaler().fit(dframe[column].values.reshape(-1,1))
      data_norm = pd.DataFrame(data_norm_fit.transform(dframe[column].values.reshape(-1,1)), 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
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
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
  for column in dframe.columns:
    try:
      
      data_norm_fit = RobustScaler().fit(dframe[column].values.reshape(-1,1))
      data_norm = pd.DataFrame(data_norm_fit.transform(dframe[column].values.reshape(-1,1)), 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))
      
      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
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
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
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

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
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
  pwrTransform_name=[]
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]
  
  for column in dframe.columns:
    try:
      
      if min(dframe[column].values) <= 0 :
        pwrTransform = PowerTransformer(method="yeo-johnson", standardize = False)
        pwrTransform_name.append("pwrTransform_y")
      else:
        pwrTransform = PowerTransformer(method="box-cox", standardize = False)
        pwrTransform_name.append("pwrTransform_b")

      data_norm_fit_transform = pwrTransform.fit_transform(dframe[column].values.reshape(-1,1))
      
      data_norm = pd.DataFrame(data_norm_fit_transform, 
                           index = dframe[column].index, 
                           columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
                      },
                      index= list( map(lambda el, name: f"{el}_{name}", dframe.columns, pwrTransform_name))
                      )

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
  mean_lst=[]
  mode_lst=[]
  median_lst=[]
  skew_lst=[]

  for column in dframe.columns:
    try:
      
      data_norm_fit = normalize([dframe[column].values])

      data_norm = pd.DataFrame(data_norm_fit[0], 
                               index = dframe[column].index,
                               columns = dframe[[column]].columns)
    
      pvalue_jarque_bera.append(round(stats.jarque_bera(data_norm[column]).pvalue,5))

      stat, p_val = stats.normaltest(data_norm[column])
      pvalue_pearson.append(round(p_val,5))

      v_ariance = round(variance(data_norm[column]),5)
      variance_lst.append(v_ariance)

      m_ean = round(data_norm[column].mean(),5)
      mean_lst.append(m_ean)

      m_ode = round(data_norm[column].mode().values[0],5)
      mode_lst.append(m_ode)

      m_edian = round(data_norm[column].median(),5)
      median_lst.append(m_edian)

      s_kew = round(data_norm[column].skew(),5)
      skew_lst.append(s_kew)
      
    except KeyError:
      print("нулевые и отрицательные значения не поддерживаются")
  
  
  df_res = pd.DataFrame({"pvalue_jarque_bera": pvalue_jarque_bera,
                      "pvalue_pearson": pvalue_pearson,
                      "variance": variance_lst,
                      "mean":mean_lst,
                      "mode":mode_lst,
                      "median":median_lst,
                      "skew":skew_lst
                      }, index=[f"{el}_normalize" for el in dframe.columns])

  df_summary = df_summary.append(df_res)


  return df_summary

# transform_collection(df_pandas)



def pvalue_test_collection(dframe, alpha=0.05, test_name=["jarque_bera"]):
  """
  Пример:
  import pandas as pd

  df_pandas = pd.DataFrame({"A":np.random.poisson(5,200),
                          "B":np.random.poisson(5,200),
                          "C":np.random.poisson(5,200)
                          })
                          
  p_res = pvalue_test_collection( df_pandas, test_name=["shapiro","ks_test","jarque_bera","pearson"])
  p_res
  """

  from scipy import stats

  df_result= pd.DataFrame()


# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::


  if "jarque_bera" in test_name:

    tn = "jarque_bera"

    if len(dframe.columns) == 1:

      jrqber = stats.jarque_bera(dframe[dframe.columns[0]]).pvalue
    
      ser_test = pd.Series(jrqber, index=[f"{dframe.columns[0]}"] ).astype("float")

      norm_res = ["Gaussian" if jrqber > alpha else "not match Gaussian"]

      df_pval = pd.DataFrame({dframe.columns[0]:ser_test.values,
                              "alpha_result":norm_res},
                             index=[tn],
                             )
      df_pval = df_pval[dframe.columns[0]].astype("object")
        # Попытка устранить ошибку:
        # 
        # /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:53: 
        # DeprecationWarning: The default dtype for empty Series will be 
        # 'object' instead of 'float64' in a future version. 
        # Specify a dtype explicitly to silence this warning.
      
      if len(df_result.columns) > 0:
        df_result = df_result.append(df_pval)
      else:
        df_result = df_result.join(df_pval, how="right")

    elif len(dframe.columns) > 1:

      jrqber_lst=[]
      df_pval=pd.DataFrame()

      for column in dframe.columns:
        jrqber = stats.jarque_bera(dframe[column]).pvalue
        jrqber_lst.append(jrqber)

      norm_res = list(map(lambda el:"Gaussian" if el > alpha else "not match Gaussian",jrqber_lst))
    
      ser_test = pd.Series(jrqber_lst, index=dframe.columns, name = tn ).astype("float")
      ser_res = pd.Series(norm_res, index=dframe.columns, name="alpha_jarque" ).astype("object")
      ser_nul = pd.Series(index=dframe.columns,name="::-1-::",dtype=pd.StringDtype()).astype("object")
        # Устранение ошибки:
        # 
        # /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:53: 
        # DeprecationWarning: The default dtype for empty Series will be 
        # 'object' instead of 'float64' in a future version. 
        # Specify a dtype explicitly to silence this warning.
        # 
        # dtype=pd.StringDtype()

      df_pval = ser_nul.to_frame().join(ser_test).join(ser_res).fillna("")
      df_result = df_result.join(df_pval, how="right").fillna(" ")


# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::


  if "pearson" in test_name:

    tn = "pearson"

    if len(dframe.columns) == 1:

      
      stat, p_value = stats.normaltest(dframe[dframe.columns[0]])
    
      ser_test = pd.Series(round(p_value,5), index=[f"{dframe.columns[0]}"] ).astype("float")

      norm_res = ["Gaussian" if p_value > alpha else "not match Gaussian"]

      df_pval = pd.DataFrame({dframe.columns[0]:ser_test.values,
                                "alpha_result":norm_res},
                               index=[tn],
                               )
      df_pval = df_pval[dframe.columns[0]].astype("object")
      
      if len(df_result.columns) > 0:
        df_result = df_result.append(df_pval)
      else:
        df_result = df_result.join(df_pval, how="right")

    elif len(dframe.columns) > 1:

      ks_test_lst=[]
      df_pval=pd.DataFrame()

      for column in dframe.columns:
        stat, p_value = stats.normaltest(dframe[column])
        ks_test_lst.append(p_value)


      norm_res = list(map(lambda el:"Gaussian" if el > alpha else "not match Gaussian",ks_test_lst))
    
      ser_test = pd.Series(ks_test_lst, index=dframe.columns, name = tn ).astype("float")
      ser_res = pd.Series(norm_res, index=dframe.columns, name="alpha_pears" ).astype("object")
      ser_nul = pd.Series(index=dframe.columns,name="::-2-::",dtype=pd.StringDtype()).astype("object")

      df_pval = ser_nul.to_frame().join(ser_test).join(ser_res).fillna("")
      df_result = df_result.join(df_pval, how="right").fillna(" ")


# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::


  if "ks_test" in test_name:
    
    # from scipy.stats import kstest

    tn = "ks_test"

    if len(dframe.columns) == 1:

      mean_val = dframe[dframe.columns[0]].mean()  # Рассчитать среднее
      std_val = dframe[dframe.columns[0]].std()  # Рассчитать стандартное отклонение

      ks_test = stats.kstest(dframe[dframe.columns[0]], 'norm',(mean_val,std_val))
    
      ser_test = pd.Series(round(ks_test[1],5), index=[f"{dframe.columns[0]}"] ).astype("float")

      norm_res = ["Gaussian" if ks_test[1] > alpha else "not match Gaussian"]

      df_pval = pd.DataFrame({dframe.columns[0]:ser_test.values,
                              "alpha_result":norm_res},
                             index=[tn],
                             )
      df_pval = df_pval[dframe.columns[0]].astype("object")
      
      if len(df_result.columns) > 0:
        df_result = df_result.append(df_pval)
      else:
        df_result = df_result.join(df_pval, how="right")

    elif len(dframe.columns) > 1:

      ks_test_lst=[]
      df_pval=pd.DataFrame()

      for column in dframe.columns:

        mean_val = dframe[column].mean()  # Рассчитать среднее
        std_val = dframe[column].std()  # Рассчитать стандартное отклонение

        ks_test = stats.kstest(dframe[column], 'norm',(mean_val,std_val))
        ks_test_lst.append(ks_test[1])


      norm_res = list(map(lambda el:"Gaussian" if el > alpha else "not match Gaussian",ks_test_lst))
    
      ser_test = pd.Series(ks_test_lst, index=dframe.columns, name = tn ).astype("float")
      ser_res = pd.Series(norm_res, index=dframe.columns, name="alpha_ks" ).astype("object")
      ser_nul = pd.Series(index=dframe.columns,name="::-3-::",dtype=pd.StringDtype()).astype("object")

      df_pval = ser_nul.to_frame().join(ser_test).join(ser_res).fillna("")
      df_result = df_result.join(df_pval, how="right").fillna(" ")


# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::


  if "shapiro" in test_name:

    import scipy

    tn = "shapiro"

    if len(dframe.columns) == 1:

      stat, p_value = stats.shapiro(dframe[dframe.columns[0]]) # тест Шапиро-Уилк
    
      ser_test = pd.Series(round(p_value,5), index=[f"{dframe.columns[0]}"] ).astype("float")

      norm_res = ["Gaussian" if p_value > alpha else "not match Gaussian"]

      df_pval = pd.DataFrame({dframe.columns[0]:ser_test.values,
                                "alpha_result":norm_res},
                               index=[tn],
                               )
      df_pval = df_pval[dframe.columns[0]].astype("object")
      
      if len(df_result.columns) > 0:
        df_result = df_result.append(df_pval)
      else:
        df_result = df_result.join(df_pval, how="right")

    elif len(dframe.columns) > 1:

      ks_test_lst=[]
      df_pval=pd.DataFrame()

      for column in dframe.columns:
        stat, p_value = stats.normaltest(dframe[column])
        ks_test_lst.append(p_value)


      norm_res = list(map(lambda el:"Gaussian" if el > alpha else "not match Gaussian",ks_test_lst))
    
      ser_test = pd.Series(ks_test_lst, index=dframe.columns, name = tn ).astype("float")
      ser_res = pd.Series(norm_res, index=dframe.columns, name="alpha_shapiro" ).astype("object")
      ser_nul = pd.Series(index=dframe.columns,name="::-4-::",dtype=pd.StringDtype()).astype("object")

      df_pval = ser_nul.to_frame().join(ser_test).join(ser_res).fillna("")
      df_result = df_result.join(df_pval, how="right").fillna(" ")
  
  return df_result
