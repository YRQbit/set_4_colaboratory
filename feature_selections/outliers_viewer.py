def qq_outliers_4_all(dframe,lst_columns=[],return_lst="no"):
  """
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
