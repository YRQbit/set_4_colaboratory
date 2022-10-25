import pandas as pd
import numpy as np

from outliers_viewer import scope_outliers_filter

def rough_filter(df_features,
                     df_target=pd.DataFrame(), 
                     var_max="mean", 
                     mean_max="mean", 
                     skew_max="mean",  
                     returned="result_table"):

  """
  # 
  # 
  rough_filter(df_features,
                     df_target=pd.DataFrame(), 
                     var_max="mean", 
                     mean_max="mean", 
                     skew_max="mean",  
                     returned="result_table")

  # rough_filter(df_pandas_scaler[["A","B","C"]],df_target=df_pandas_scaler[["D"]])

  # rough_filter(df_pandas_scaler)

  # rough_filter(df_pandas_scaler, var_max=10, returned="result_table_s")

  # rough_filter(df_pandas_scaler, mean_max=15, returned="complience_table_s")

  # rough_filter(df_pandas_scaler, var_max=5, skew_max=0.5, returned="correlation_table_s")

  # rough_filter(df_pandas, returned="result_table")

  # rough_filter(df_pandas, returned="complience_table")

  # rough_filter(df_pandas, returned="correlation_table")
  """

  df_std = pd.DataFrame()
  vari_value = {}
  std_value = {}
  mean_value={}
  skew_value = {}

# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

  # Вычисление показателей корреляции
  # 
  if (len(df_target.columns) != 0) and (len(df_target.index) != 0):
    # 
    # В случае, когда Таргет указан явно
    # .
    target_name = df_target.columns[0]
    
    df_correlation = df_features.join(df_target, how="left", lsuffix=target_name,rsuffix='')
    df_correlation = df_correlation.corr()
    
  else:

    # В случае, когда Таргет не указан явно, используем 
    # в качестве Таргета последний столбец
    # .
    df_correlation = df_features.corr()

    df_target = df_correlation.loc[:, df_correlation.columns[-1]].to_frame()
    target_name = df_target.columns[0]
    
    df_features = df_features.loc[:, :df_features.columns[-2]]

# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

  df_scope_out = scope_outliers_filter(df_features)[["column_name","scope","outliers_count"]]

  # Наполнение словарей
  # 
  for el in df_features.columns:
    # .
    std_value[el] = df_features[el].std().round(3)
    skew_value[el] = df_features[el].skew().round(3)
    vari_value[el] = df_features[el].var().round(3)
    mean_value[el] = df_features[el].mean().round(3)

    next
  

  # Добавление инфо-строк в df_features
  # 
  df_std = df_std.append(vari_value, ignore_index=True)\
                 .append(std_value, ignore_index=True)\
                 .append(mean_value, ignore_index=True)\
                 .append(skew_value, ignore_index=True)
  df_std = df_std.append(df_scope_out.set_index("column_name").T)
  df_std.rename(index={0: "variance", 1: "std", 2: "mean", 3: "skew"}, inplace=True)
  
  means_dic = {"variance_mean": df_std.loc["variance"].mean().round(5),
               "std_mean": df_std.loc["std"].mean().round(5),
               "mean_mean": df_std.loc["mean"].mean().round(5),
               "skew_mean": df_std.loc["skew"].mean().round(5),
               "scope_mean": df_std.loc["scope"].mean().round(5),
               "outliers_count_mean":df_std.loc["outliers_count"].mean().round(5)
               }
  
  df_std.insert(0, "means", means_dic.values(), True)


  if var_max == "mean": var_max = means_dic["variance_mean"]
  if mean_max == "mean": mean_max = means_dic["mean_mean"]
  if skew_max == "mean": skew_max = means_dic["skew_mean"]


  df_std = df_std.append(df_correlation[df_features.columns].iloc[-1].rename("corr"))

  # row of scores 
  # 
  df_std = df_std.append(pd.Series([0 for el in df_std.columns], name="scores", index=df_std.columns))

# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ
  
  for column in df_std.columns:
    if column == "means": df_std[column].loc["scores"] = np.nan
    
    if df_std[column].loc["variance"] > var_max:
      df_std[column].loc["scores"] += 1

    if df_std[column].loc["std"] < means_dic["std_mean"]: 
      df_std[column].loc["scores"] += 1
    
    if df_std[column].loc["mean"] > mean_max:
      df_std[column].loc["scores"] += 1

    if (abs(df_std[column].loc["skew"]) <= abs(skew_max)) and(abs(skew_max) <= 1 ): 
      df_std[column].loc["scores"] += 1

    if (df_std[column].loc["scope"] > means_dic["scope_mean"])\
    and(df_std[column].loc["outliers_count"] == 0):
      df_std[column].loc["scores"] += 3

    if (df_std[column].loc["scope"] < means_dic["scope_mean"])\
    and(df_std[column].loc["outliers_count"] == 0):
      df_std[column].loc["scores"] += 2
    
    elif (df_std[column].loc["scope"] < means_dic["scope_mean"])\
    and(df_std[column].loc["outliers_count"] < means_dic["outliers_count_mean"]):
      df_std[column].loc["scores"] += 1

    if (df_std[column].loc["scope"] > means_dic["scope_mean"])\
    and(df_std[column].loc["outliers_count"] < means_dic["outliers_count_mean"]):
      df_std[column].loc["scores"] += 1

    if (df_std[column].loc["scope"] < means_dic["scope_mean"])\
    and(df_std[column].loc["outliers_count"] > means_dic["outliers_count_mean"]):
      df_std[column].loc["scores"] -= 1


  corr_score = {0: 0.1,
                1: 0.2,
                2: 0.3,
                3: 0.4,
                4: 0.5,
                5: 0.6,
                6: 0.7,
                7: 0.8,
                8: 0.9,
                9: 1.0}

  corr_val_lst = [abs(el) for el in df_std.iloc[6,1:] ]
  corr_score_keys = list(corr_score.keys())
  scorr_lst=[]

  for usval in corr_val_lst:

    for k, keyval in enumerate(corr_score.values()):
      
      if (corr_score_keys[k] == 0) and (usval <= corr_score.get(0)):
        scorr_lst.append(corr_score_keys[k])

      elif (corr_score_keys[k] == 9) and ( usval >= corr_score.get(9)):
        scorr_lst.append(corr_score_keys[0])
    
      else:
        xcv = corr_score.get(k+1)
        if (usval >= keyval) and (usval < xcv): 
          scorr_lst.append(corr_score_keys[k])


  scorr_lst = list(map(sum,zip(df_std.iloc[7,1:],scorr_lst)))

  df_std.iloc[7,1:] = pd.Series([el for el in scorr_lst])

  df_std = df_std.fillna("")

# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

  def result_table(dframe,s="y"):

    if s == "y":
      s = dframe.style.set_caption("rough filter :: result_table").set_table_styles([{
        'selector': 'caption',
        'props': 'caption-side: header; font-size:1.25em; background-color: lightyellow'
        }], overwrite=False)

      return s
    
    else:
      return dframe

  def complience_table(dframe,s="y"):
    
    mean_val = dframe.loc["scores"][1:].mean() #.count()

    complience = dframe.loc["scores"][1:].sort_values(ascending=False)

    complience_table = complience[:(complience >= mean_val).sum()].to_frame()

    if s == "y":

      s = complience_table.style.set_caption("rough filter :: complience_table").set_table_styles([{
        'selector': 'caption',
        'props': 'caption-side: header; font-size:1.25em; background-color: lightyellow'
        }], overwrite=False)

      return s
    
    else:
      return complience_table

  def correlation_table(dframe,s="y"):

    if s == "y":
      s = dframe.style.format('{:.5f}')
      s.set_caption("rough filter :: correlation_table").set_table_styles([{
          'selector': 'caption',
         'props': 'caption-side: header; font-size:1.25em; background-color: lightyellow'
          }], overwrite=False)

      return s

    else: 
      return dframe

# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

  result = {
  'result_table_s': result_table(df_std), #lambda x: x * 5,
  'complience_table_s': complience_table(df_std), # lambda x: x + 7,
  'correlation_table_s': correlation_table(df_correlation), # lambda x: x - 2
  'correlation_table': correlation_table(df_correlation,s="n"),
  'complience_table': complience_table(df_std,s="n"),
  'result_table': result_table(df_std,s="n")
  }[returned] #(x)

  return result


# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::
# ::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::-::



def pvalue_test_collection(dframe, alpha=0.05, test_name=["jarque_bera"]):
  """
  #
  # .
  Пример:
  
  import pandas as pd

  df_pandas = pd.DataFrame({"A":np.random.poisson(5,200),
                          "B":np.random.poisson(5,200),
                          "C":np.random.poisson(5,200)
                          })
                          
  p_res = pvalue_test_collection( df_pandas, 
                                  test_name=["shapiro",
                                             "ks_test",
                                             "jarque_bera",
                                             "pearson"])
  p_res
  """

  from scipy import stats

  df_result= pd.DataFrame()


# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ


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


# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ


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


# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ


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


# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ


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


def SelectKBest_f_regression(X_dframe,Y_series, n_best=2):
  """
  :
  :
  skbfr = SelectKBest_f_regression(X_dframe=X_Cols,
                         Y_series=Y_Target, 
                         n_best=5)
  
  skbfr.to_frame().rename(columns = {0:"SKB_fReg"})
  """
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import f_regression

  
  KBest = SelectKBest(score_func = f_regression, k = n_best)
  KBest = KBest.fit(X_dframe,Y_series)

  cols = KBest.get_support(indices=True)

  features_lst = X_dframe.columns[cols]
  
  return features_lst



def SelectKBest_chi2(X_dframe,Y_series, n_best=2):
  """
  :
  :
  skbx2 = SelectKBest_chi2(X_dframe=X_Cols,
                         Y_series=Y_Target, 
                         n_best=5)
  
  skbx2.to_frame().rename(columns = {0:"SKB_chi2"})
  """
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2

  KBest = SelectKBest(score_func = chi2, k = 5)
  KBest = KBest.fit(X_dframe,Y_series)

  cols = KBest.get_support(indices=True)

  features_lst = X_dframe.columns[cols]
  
  return features_lst


def SelectPercentile_f_regression(X_dframe,Y_series, percents=80):
  """
  :
  :
  percents=80 :: выборка 80% лучших фич на основе их оценок
  
  SelectPercentile_f_regression(X_dframe=X_Cols,
                         Y_series=Y_Target, 
                         percents=50)
  """
  from sklearn.feature_selection import SelectPercentile
  from sklearn.feature_selection import f_regression


  SPercentile = SelectPercentile(score_func = f_regression, percentile=percents)
    
  SPercentile = SPercentile.fit(X_dframe,Y_series)

  cols = SPercentile.get_support(indices=True)

  features_lst = X_dframe.columns[cols]
  
  return features_lst



def SelectPercentile_chi2(X_dframe,Y_series, percents=80):
  """
  :
  :
  percents=80 :: выборка 80% лучших фич на основе их оценок
  
  SelectPercentile_chi2(X_dframe=X_Cols,
                         Y_series=Y_Target, 
                         percents=50)
  """
  from sklearn.feature_selection import SelectPercentile
  from sklearn.feature_selection import chi2


  SPercentile = SelectPercentile(score_func = chi2, percentile=percents)
    
  SPercentile = SPercentile.fit(X_dframe,Y_series)

  cols = SPercentile.get_support(indices=True)

  features_lst = X_dframe.columns[cols]
  
  return features_lst



def SFM_RFReg(X_dframe,Y_series, n_estimators_=20):
  """
  :
  :
  sfmrfreg = SFM_RFReg(X_dframe=X_Cols,
          Y_series=Y_Target,
          n_estimators_=20)

  pd.DataFrame({"SFM_RFReg": sfmrfreg},
             index=sfmrfreg)
  """
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.feature_selection import SelectFromModel

  regressor = RandomForestRegressor(n_estimators=n_estimators_,
                                    random_state=52)
  
  reg_fit = SelectFromModel(estimator=regressor)\
            .fit(X_dframe,Y_series)\
            .get_feature_names_out(X_dframe.columns)
  
  return reg_fit



def SFM_RFClass(X_dframe,Y_series, n_estimators_=20):
  """
  :
  :
  sfmrfclass = SFM_RFClass(X_dframe=X_Cols,
          Y_series=Y_Target,
          n_estimators_=20)

  pd.DataFrame({"SFM_RFClass": sfmrfclass},
             index=sfmrfclass)
  """
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.feature_selection import SelectFromModel

  regressor = RandomForestClassifier(n_estimators=n_estimators_,
                                     random_state=52)
  
  reg_fit = SelectFromModel(estimator=regressor)\
            .fit(X_dframe,Y_series)\
            .get_feature_names_out(X_dframe.columns)
  
  return reg_fit


def SFM_LReg(X_dframe,Y_series):
  """
  :
  :
  sfmlreg = SFM_LReg(X_dframe=X_Cols,
         Y_series=Y_Target
         )
         
  pd.DataFrame({"SFM_LReg": sfmlreg},
             index=sfmlreg)
  """

  from sklearn.linear_model import LinearRegression
  from sklearn.feature_selection import SelectFromModel 

  regressor = LinearRegression()
  reg_fit = SelectFromModel(estimator=regressor)\
            .fit(X_dframe,Y_series)\
            .get_feature_names_out(X_dframe.columns)

  return reg_fit


def SFM_SGDReg(X_dframe,Y_series, n_tol=0.0001, n_eta=0.01):
  """
  :
  :
  sfmsgdreg = SFM_SGDReg(X_dframe=X_Cols,
                         Y_series=Y_Target
                        )
                        
  pd.DataFrame({"SFM_LReg": sfmsgdreg},
             index=sfmsgdreg)
  """
  
  from sklearn.linear_model import SGDRegressor
  from sklearn.feature_selection import SelectFromModel

  regressor = SGDRegressor(tol=n_tol,
                           eta0=n_eta, random_state=52 )
  
  reg_fit = SelectFromModel(estimator=regressor)\
            .fit(X_dframe,Y_series)\
            .get_feature_names_out(X_dframe.columns)

  return reg_fit


def RFE_SVR(X_dframe,Y_series, kernel_="linear",n_features_to_select_=2, step_=1):
  """
  :
  :
  rfesvr = RFE_SVR(X_dframe=X_Cols,
        Y_series=Y_Target,
        n_features_to_select_=5
        )
        
  pd.DataFrame({"RFE_SVR": rfesvr},
             index=rfesvr)
  """

  from sklearn.feature_selection import RFE
  from sklearn.svm import SVR

  estimator = SVR(kernel=kernel_)
  selector = RFE(estimator,
                 n_features_to_select=n_features_to_select_,
                 step=step_)
  selector = selector.fit(X_dframe, Y_series)
  
  # selector.support_
  # selector.ranking_

  cols = selector.get_support(indices=True)
  features = X_dframe.columns[cols]

  return features


def RFE_LassoCV(X_dframe,Y_series, n_features_to_select_=2, step_=1):
  """
  rfelassocv = RFE_LassoCV(X_dframe=X_Cols,
            Y_series=Y_Target,
            n_features_to_select_=5
        )

  pd.DataFrame({"RFE_LassoCV": rfelassocv},
             index=rfelassocv)
  """

  from sklearn.feature_selection import RFE
  from sklearn.linear_model import LassoCV

  estimator = LassoCV()

  selector = RFE(estimator,
                 n_features_to_select=n_features_to_select_,
                 step=step_)
  selector = selector.fit(X_dframe, Y_series)
  
  # selector.support_
  # selector.ranking_

  cols = selector.get_support(indices=True)
  features = X_dframe.columns[cols]

  return features


def RFE_LReg(X_dframe,Y_series, n_features_to_select_=2, step_=1):
  """
  rfelreg = RFE_LReg(X_dframe=X_Cols,
            Y_series=Y_Target,
            n_features_to_select_=5
        )
        
  pd.DataFrame({"RFE_LReg": rfelreg},
             index=rfelreg)
  """

  from sklearn.feature_selection import RFE
  from sklearn.linear_model import LinearRegression

  estimator = LinearRegression()

  selector = RFE(estimator,
                 n_features_to_select=n_features_to_select_,
                 step=step_)
  selector = selector.fit(X_dframe, Y_series)
  
  # selector.support_
  # selector.ranking_

  cols = selector.get_support(indices=True)
  features = X_dframe.columns[cols]

  return features


def SFS_LassoCV(X_dframe,Y_series, n_features_to_select_=2, direction_="forward"):
  """
  :
  :
  SFS_LassoCV(X_dframe=X_Cols,
         Y_series=Y_Target,
         n_features_to_select_=5,
         direction_="forward"
        )
  :
  :
  SFS_LassoCV(X_dframe=X_Cols,
         Y_series=Y_Target,
         n_features_to_select_=5,
         direction_="backward"
        )
  """

  from sklearn.feature_selection import SequentialFeatureSelector
  from sklearn.linear_model import LassoCV

  estimator = LassoCV()

  selector = SequentialFeatureSelector(estimator, 
                                       n_features_to_select=n_features_to_select_,
                                       direction=direction_)\
                                       .fit(X_dframe, Y_series)

  cols = selector.get_support()

  features_lst = X_dframe.columns[cols]


  return features_lst



def SFS_RFReg(X_dframe,Y_series, n_features_to_select_=2, direction_="forward"):
  """
  :
  :
  SFS_RFReg(X_dframe=X_Cols,
          Y_series=Y_Target,
          n_features_to_select_=5,
          direction_="forward")
  :
  :
  SFS_RFReg(X_dframe=X_Cols,
          Y_series=Y_Target,
          n_features_to_select_=5,
          direction_="backward")
  """
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.feature_selection import SequentialFeatureSelector

  estimator = RandomForestRegressor(n_estimators=20, random_state=52)
  
  selector = SequentialFeatureSelector(estimator, 
                                       n_features_to_select=n_features_to_select_,
                                       direction=direction_)\
                                       .fit(X_dframe, Y_series)

  cols = selector.get_support()

  features_lst = X_dframe.columns[cols]
  
  return features_lst



def coef_ElNet(X_dframe,Y_series,n_features_to_select_=2, plot="yes",abs_coef_ = "yes",**kwargs):
  """
  :
  :
  coef_ElNet(X_dframe=X_Cols,
           Y_series=Y_Target,
           n_features_to_select_=5,
           abs_coef_ = "yes",
           positive=False,
           )
  :
  :
  # ElasticNet_attr:
  # 
  # random_state=52,
  # abs_coef_ = "yes"
  # alpha=0.001,
  # l1_ratio=0.75,
  # normalize=True,
  # positive=True
  :
  :
  df_coef_ElNet = coef_ElNet(X_dframe=X_Cols,
          Y_series=Y_Target,
          n_features_to_select_=5)
  
  df_coef_ElNet["feature"].to_list()
  """
  import matplotlib.pyplot as plt
  from sklearn.linear_model import ElasticNet

  estimator=ElasticNet()


  if len(kwargs) != 0:
    for var_name, var_value in kwargs.items():
      if hasattr(estimator,var_name) == True:
        setattr(estimator, var_name, var_value)
      else:
        print(f"\033[35m::Warning::\033[0m ElasticNet does not contain an attribute :: \033[35m{var_name}\033[0m")


  selector = estimator.fit(X_dframe,Y_series)


  if abs_coef_ == "yes": importance = np.abs(selector.coef_)
  elif abs_coef_ == "no": importance = selector.coef_


  feature_names = np.array(X_dframe.columns)

  df_coef = pd.DataFrame({"feature": feature_names,"coef_": importance})\
                          .sort_values(["coef_"],ascending=False)\
                          .head(n_features_to_select_)

  if plot == "yes":

    plt.barh(width=df_coef["coef_"], y=df_coef["feature"]),
    plt.title("Feature importances via coefficients ElasticNet")
    plt.show()
    print("\n","::-::-"*15+"::"+"\n")

  
  return df_coef

  features_lst = X_dframe.columns[cols]
  
  return reg_fit
