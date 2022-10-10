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
      
