import pandas as pd
import numpy as np

def qq_outliers_dframe(dframe,lst_columns=[],return_lst="no"):
  """
  :
  :
  Назначение: IRQ-обнаружение выбросов по всем переменным

  example:

  # feature_outliers = qq_outliers_4_all(dframe=sdf,lst_columns=sdf.columns,return_lst="no")
  # display(len(feature_outliers[0]))
  # display(len(feature_outliers[1]))
  # feature_outliers
  
  """

  min_idx=[]
  max_idx=[]

  for el in lst_columns:
      q75,q25 = np.percentile( dframe.loc[:,el],[75,25])
      intr_qr = q75-q25

      max = q75+(1.5*intr_qr)
      min = q25-(1.5*intr_qr)

      print(f"\033[34m{el}\033[0m :: outliers < min_::_ "+f"{len(dframe.loc[ dframe[el] < min].index)}")
      print(f"\033[34m{el}\033[0m :: outliers > max_::_ "+f"{len(dframe.loc[ dframe[el] > max].index)}")
      print("\n")

      min_idx.extend(dframe.loc[ dframe[el] < min].index)
      max_idx.extend(dframe.loc[ dframe[el] > max].index)
  
  min_idx = list(set(min_idx))
  max_idx = list(set(max_idx))

  print(f"\033[35mall outliers < min\033[0m in dframe :: {len(min_idx)}")
  print(f"\033[35mall outliers > max\033[0m in dframe :: {len(max_idx)}")
  print(f"\033[35mall outliers min + max\033[0m in dframe :: {len(min_idx) + len(max_idx)}")
  print("\n")

  if return_lst != "no": return min_idx,max_idx

  

def qq_outliers_series(ser_data):
  """
  :
  :
  Подключение:
  from outliers_viewer import qq_outliers_series
  
  На входе:
  qq_outliers_series(dframe[column])
  
  На выходе:
  outliers_dict = {"count" : count_outliiers,
                   "count_min" : len(min_idx[0]),
                   "count_max" : len(max_idx[0]),
                   "val_min_outliers" : list(min_idx[0]),
                   "val_max_outliers" : list(max_idx[0])}
  """

  min_idx=[]
  max_idx=[]

  q75,q25 = np.percentile(ser_data,
                          [75,25],
                          interpolation='midpoint')
  IQR = q75 - q25

  upper = ser_data >= (q75+1.5*IQR)
  lower = ser_data <= (q25-1.5*IQR)

  min_idx = np.where(lower)
  max_idx = np.where(upper)

  count_outliiers = len(min_idx[0])+len(max_idx[0])

  outliers_dict = {"count" : count_outliiers,
                   "count_min" : len(min_idx[0]),
                   "count_max" : len(max_idx[0]),
                   "val_min_outliers" : list(min_idx[0]),
                   "val_max_outliers" : list(max_idx[0])}
  
  return outliers_dict


def scope_outliers_filter(dframe):
  """
  :
  :
  Назначение: Создание сводной таблицы с показателями размаха 
  и количества выбросов по каждой переменной

  Пример:

  scope_outliers_filter(df_pandas)
  """

  sc_qq = []
  qq_out_count = []
  qq_out_count_min = []
  qq_out_count_max = []
  qq_out_val_min = []
  qq_out_val_max = []

  for column in dframe.columns:

    qos = qq_outliers_series(dframe[column])

    qq_out_count.append(qos["count"])

    qq_out_count_min.append(qos["count_min"])

    qq_out_count_max.append(qos["count_max"])

    qq_out_val_min.append(qos["val_min_outliers"])

    qq_out_val_max.append(qos["val_max_outliers"])

    scope = round(dframe[column].max() - dframe[column].min(),5)

    sc_qq.append(scope)

  df_scope = pd.DataFrame({"column_name" : list(dframe.columns),
                           "scope" : sc_qq,
                           "outliers_count" : qq_out_count,
                           "outliers_count_min" : qq_out_count_min,
                           "outliers_count_max" : qq_out_count_max,
                           "val_min_outliers" : qq_out_val_min,
                           "val_max_outliers" : qq_out_val_max
                           })

  return df_scope
