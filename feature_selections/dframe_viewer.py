def unique_in_df(dframe):
  """
  Назначение: список уникальных значений столбцов

  example:

  rows_val_1 = ["a","b","c"]
  rows_val_2 = ["d","e","f"]
  df_test = pd.DataFrame({1:rows_val_1,
                          2:rows_val_2})

  dframe = df_test
  unique_in_df(dframe)

  example:
  unique_in_df(dframe=df_test)

  example:
  unique_in_df(df_test)
  """
  for el in dframe.columns:
    print(f"\033[34m{el}\033[0m",f"::{dframe[el].nunique()}")
    print(dframe[el].unique())
    print()

def feature_explorer(dframe,target_name):

  # from statistics import variance

  df_target = dframe[[target_name]].copy()
  df_features = dframe.drop(columns=target_name)

  print("::-::-"*15+"::")
  print("\033[31m\033[43mSHAPE:")
  print("\033[0m")
  display(dframe.shape)
  print("\n")

  print("::-::-"*15+"::")
  print("\033[31m\033[43mUNIQUE:")
  print("\033[0m")
  unique_in_df(df_target)
  unique_in_df(df_features)
  print("\n")

  print("::-::-"*15+"::")
  print("\033[31m\033[43mMODE:")
  print("\033[0m")
  display(df_features.mode())
  print("\n")

  print("::-::-"*15+"::")
  print("\033[31m\033[43mDESCRIBE:")
  print("\033[0m")
  display(dframe.describe().round(3))
  print('\n')

  print("::-::-"*15+"::")
  print("\033[31m\033[43mVARIANCE:")
  print("\033[0m")
  display(dframe_variance(dframe))
  print('\n')

  print("::-::-"*15+"::")
  print("\033[31m\033[43mSKEW:")
  print("\033[0m")
  display(feature_skew(dframe))
  print('\n')

  print("::-::-"*15+"::")
  print("\033[31m\033[43mOUTLIERS:")
  print("\033[0m")
  qq_outliers_4_all(dframe,lst_columns=dframe.columns,return_lst="no")
  print('\n')

  print("::-::-"*15+"::")
  print("\033[31m\033[43mCORRELATION:")
  print("\033[0m")
  display(dframe.corr())
  print('\n')

  print("::-::-"*15+"::")
  print("\033[31m\033[43mDESCRIPRIONS:")
  print("\033[0m")
  print(f"exmple :: !cat /content/train.txt |grep -e '^m.'")
  print('\n')
  
  return
