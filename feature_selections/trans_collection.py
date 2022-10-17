from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np

def qqTransform(dframe):
  """
  """
  from sklearn.preprocessing import QuantileTransformer
  
  qqTransform = QuantileTransformer(output_distribution="normal", random_state = 42)

  if len(dframe.columns) < 2:
    
    qq_target = qqTransform.fit_transform(dframe[[dframe.columns[0]]])
    qq_invers = qqTransform.inverse_transform(qq_target)
    
    df_qq_transform = pd.DataFrame({f"{dframe.columns[0]}_qqT":pd.Series(list(x for el in qq_target for x in el)) })
    df_qq_invers = pd.DataFrame({f"{dframe.columns[0]}_qqInv":pd.Series(list(x for el in qq_invers for x in el)) })

    df_result = df_qq_transform.join(df_qq_invers)

    return df_result
  
  elif len(dframe.columns) >= 2:

    qq_target = qqTransform.fit_transform(dframe)
    qq_invers = qqTransform.inverse_transform(qq_target)

    df_qq_transform = pd.DataFrame(qq_target,
                                   index = dframe.index,
                                   columns = [f"{el}_qqT" for el in dframe.columns]
                                   )
    
    df_qq_invers = pd.DataFrame(qq_invers,
                                index = dframe.index,
                                columns = [f"{el}_qqInv" for el in dframe.columns]
                                )
    
    df_result = df_qq_transform.join(df_qq_invers)

    return df_result
  
  
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

  
def transform_viewer(dframe):
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

# transform_viewer(df_pandas)

