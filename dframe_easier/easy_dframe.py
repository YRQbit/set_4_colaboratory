import pandas as pd
import numpy as np

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def row_insert_one_before(drame,insert_before_index=1):
  """
  :
  import pandas as pd
  import numpy as np
  :
  sdf_4 = pd.DataFrame({"col_13": ["val_12","val_13","val_20",np.nan,"val_26"],
  :                     "col_24": ["val_14","val_15",np.nan,"val_21",np.nan],
  :                     "col_35": ["val_16",np.nan,"val_17","val_22",np.nan],
  :                     "col_52": ["val_18","val_19","val_23","val_24","val_25"]})
  :
  :
  sdf_4 = row_insert_one_before(sdf_4,insert_before_index=3)
  sdf_4
  :
  """

  df_col_count = drame.shape[0]

  new_idx_lst = [el for el in range(0,insert_before_index,1) ]\
   + [el for el in range(insert_before_index,df_col_count,1)]
  
  new_idx_lst.insert(insert_before_index,"row_ins_1")
  nan_lst = [np.nan for el in new_idx_lst]
  
  df_0 = pd.DataFrame({"riob_ins_0":nan_lst}, index=new_idx_lst)

  df_ins = df_0.join(drame)
  df_ins.reset_index(drop=True, inplace=True)
  df_ins.drop(columns=["riob_ins_0"],inplace=True)
  df_ins
  
  return df_ins

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def row_insert_one_after(drame,insert_after_index=1):
  """
  :
  import pandas as pd
  import numpy as np
  :
  sdf_4 = pd.DataFrame({"col_13": ["val_12","val_13","val_20",np.nan,"val_26"],
  :                     "col_24": ["val_14","val_15",np.nan,"val_21",np.nan],
  :                     "col_35": ["val_16",np.nan,"val_17","val_22",np.nan],
  :                     "col_52": ["val_18","val_19","val_23","val_24","val_25"]})
  :
  :
  row_insert_one_after(sdf_4,insert_after_index=3)
  sdf_4
  :
  """
  df_col_count = drame.shape[0]
  
  new_idx_lst = [el for el in range(0,insert_after_index+1,1) ] + [el for el in range(insert_after_index+1,df_col_count,1)]

  new_idx_lst.insert(insert_after_index+1,"row_ins_1")

  nan_lst = [np.nan for el in new_idx_lst]

  df_0 = pd.DataFrame({0:nan_lst}, index=new_idx_lst)

  df_ins = df_0.join(drame)
  df_ins.reset_index(inplace=True)
  df_ins.drop(columns=["index",0],inplace=True)
  df_ins
  
  return df_ins

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def col_copy_one_bigger(dframe_in,dframe_out,out_col_name=""):
  """
  :
  sdf_4 = pd.DataFrame({"col_13": ["val_12","val_13","val_20",np.nan,"val_26"],
  :                     "col_24": ["val_14","val_15",np.nan,"val_21",np.nan],
  :                     "col_35": ["val_16",np.nan,"val_17","val_22",np.nan],
  :                     "col_52": ["val_18","val_19","val_23","val_24","val_25"]})
  :  
  sdf_2 = pd.DataFrame({"col_13": ["val_12","val_13"],
  :                     "col_24": ["val_14","val_15"]},
  :                    index=[0,1])
  :
  sdf = col_copy_one_bigger(dframe_in=sdf_2,
  :                         dframe_out=sdf_4,
  :                         out_col_name="col_52")
  :  
  """
  
  count_row_df_in = dframe_in.shape[0]
  count_row_df_out = dframe_out.shape[0]

  if count_row_df_out > count_row_df_in:
    dframe_in = dframe_in\
    .reindex(list(range(0, count_row_df_out))).reset_index(drop=True)
              
  dframe_in["big_ger_ins_1"] = dframe_out[out_col_name].reset_index()[out_col_name]

  return dframe_in

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def row_add_Cols_as_Rows(dframe_out=pd.DataFrame(),
                         out_col_name=[],
                         dframe_in=pd.DataFrame(),
                         orig_name=True):
  """
  Тут есть над чем поработать, см. комментарии DFRame_Class_dev.
  :
  sdf_1 = pd.DataFrame({"col_1": ["val_1"],
  :                     "col_2": ["val_1"]})
  :
  sdf_2 = pd.DataFrame({"col_13": ["val_12","val_13"],
  :                     "col_24": ["val_14","val_15"],
  :                     "col_35": ["val_18","val_17"]})
  :
  row_add_Cols_as_Rows(dframe_out=sdf_2,
  :                    out_col_name=sdf_2.columns,
  :                    dframe_in=sdf_1)
  :
  row_add_Cols_as_Rows(dframe_out=sdf_2,
  :                    out_col_name=["col_35", "col_13","col_24"],
  :                    dframe_in=sdf_1)
  :
  sdf_3 = pd.DataFrame({"col_1": ["val_1"],
  :                     "col_2": ["val_1"],
  :                     "col_3": ["val_1"],
  :                     "col_4": ["val_1"]})
  :
  sdf_4 = pd.DataFrame({"col_13": ["val_12","val_13"],
  :                     "col_24": ["val_14","val_15"],
  :                     "col_35": ["val_16","val_17"],
  :                     "col_52": ["val_18","val_19"]})
  :
  row_add_Cols_as_Rows(dframe_out=sdf_4,
  :                    out_col_name=["col_24", "col_35"],
  :                    dframe_in=sdf_3)
  :
  :
  :
  sdf_3 = pd.DataFrame({"col_1": ["val_1"],
  :                     "col_2": ["val_1"],
  :                     "col_3": ["val_1"],
  :                     "col_4": ["val_1"]})
  :
  sdf_4 = pd.DataFrame({"col_13": ["val_12","val_13","val_20",np.nan,"val_26"],
  :                     "col_24": ["val_14","val_15","val_21",np.nan,np.nan],
  :                     "col_35": ["val_16","val_17","val_22",np.nan,np.nan],
  :                     "col_52": ["val_18","val_19","val_23","val_24","val_25"]})
  :
  row_add_Cols_as_Rows(dframe_out=sdf_4,
  :                    out_col_name=["col_52", "col_13","col_24"],
  :                    dframe_in=sdf_3)
  :
  row_add_Cols_as_Rows(dframe_out=sdf_4,
  :                    out_col_name=["col_35"],
  :                    dframe_in=sdf_3)
  :    
  """
  
  lst_col_name = dframe_in.columns
  rows_nd_array = np.array(dframe_out.loc[:,out_col_name]) #.to_list()

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>

  if len(rows_nd_array) > dframe_in.shape[1]:
    # dframe_out.shape = (row=4,...)
    # dframe_in.shape = (...,col=3)
    
    print("\n conditional_1 \n")

    diff_count = len(rows_nd_array) - dframe_in.shape[1]
    
    for q in range(0,diff_count):
      dframe_in[f"dfr_{q}"] = np.nan

    for k, column in enumerate(out_col_name):
      
      diff_count = dframe_in.shape[1]-len(dframe_out[column])
      nan_vals = [np.nan for el in range(0,diff_count,1)]
      
    
      exz_lst = dframe_out[column].to_list()
      exz_lst.extend(nan_vals)

      dframe_in.loc[f"row_{k}",:] = pd.Series(exz_lst,index=dframe_in.columns)
      
      if orig_name == True:

        dframe_in.rename(index={lst_col_name[k]:out_col_name[k]})

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
  
  elif (len(rows_nd_array) == dframe_in.shape[1]) and (len(out_col_name) != 1):
    # dframe_out.shape = (row=3,...) + out_col_name = ["col_1","col_2"]
    # dframe_in.shape = (...,col=3)

    print("\n conditional_2 \n")
    
    for k, column in enumerate(out_col_name):

      exz_lst = dframe_out[column].to_list()

      dframe_in.loc[f"row_{k}",:] = pd.Series(exz_lst,index=dframe_in.columns)

      if orig_name == True:
       
        dframe_in = dframe_in.rename(index={f"row_{k}":out_col_name[k]})

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
  
  elif (len(rows_nd_array) < dframe_in.shape[1]) and (len(out_col_name) != 1):
    # dframe_out.shape[0] = (row=3,...) + len(out_col_name) == 2
    # dframe_in.shape[1] = (...,col=4)

    print("\n conditional_3 \n")

    for k, column in enumerate(out_col_name):
      
      diff_count = dframe_in.shape[1]-len(dframe_out[column])
      nan_vals = [np.nan for el in range(0,diff_count,1)]
      
      exz_lst = dframe_out[column].to_list()
      exz_lst.extend(nan_vals)

      dframe_in.loc[f"row_{k}",:] = pd.Series(exz_lst,index=dframe_in.columns)

      if orig_name == True:
        
        dframe_in = dframe_in.rename(index={f"row_{k}":out_col_name[k]})

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>

  elif (len(rows_nd_array[0]) < dframe_out.shape[1] )and(len(out_col_name) == 1):
    # dframe_out.shape = (...,col=3) + len(out_col_name) == 1 + rows_nd_array[0]== 1..N
    # dframe_in.shape = (...,col=4)

    print("\n conditional_4 \n")
    
    diff_count = dframe_in.shape[1]-len(dframe_out[out_col_name].values)
    
    nan_vals = [np.nan for el in range(0,diff_count,1)]

    exz_lst = dframe_out[out_col_name[0]].to_list()
    exz_lst.extend(nan_vals)

    dframe_in.loc[f"row_{0}",:] = pd.Series(exz_lst,index=dframe_in.columns)
      
    if orig_name == True:
               
      dframe_in = dframe_in.rename(index={f"row_{0}":out_col_name[-1]})
    
  return dframe_in

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def row_add_one(dframe,fillin = "nan"):
  """
  :
  dframe_add_one_row(X_Cols)
  :
  row_add_one(df_components).fillna("").rename(index={13:""})
  :
  """
  if fillin == "nan": fillin = np.NaN
  
  n_columns = dframe.shape[1]
  lst_values = [fillin for el in range(0,n_columns,1)]
  
  pd_ser = pd.Series(lst_values)
  dframe.loc[len(dframe.index)] = pd.Series(lst_values)
  
  return dframe

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def row_add_any(dframe,count_rows=1, fillin = "nan"):
  if fillin == "nan":
    dframe = dframe.reindex(list(range(0,dframe.shape[0] + count_rows,1) )).reset_index(drop=True)
    return dframe
  else:
    # нужно заменить значения ячеек в созданных строках
    dframe = dframe.reindex(list(range(0,dframe.shape[0] + count_rows,1) ))\
    .reset_index(drop=True)
    return dframe.fillna(fillin)
  
#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def row_rm_one(dframe,index_number="all"):
  """
  :
  dframe_row_rm_one(X_Cols,index_number=-1)
  X_Cols
  :
  """
  try:
    if index_number == "all":
      print("Нужно указать номер \ наименование индекса строки")
      print("Выполнение завершено.")
      return
    else:
      dframe = dframe.drop(index=[index_number])

      return dframe
  except (KeyError):
    print(f"{index_number} не обнаружено")
    return dframe

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def row_rm_any(dframe,index_number=[]):
  """
  :
  dframe_row_rm_any(X_Cols,index_number=[509,510])
  X_Cols
  :
  """
  
  try:
    if index_number == "all":
      print("Нужно указать номер \ список \ наименование индексов(а) строк")
      print("Выполнение завершено.")
      return
    else:
      dframe = dframe.drop(index=index_number)
        # inplace=True
      return dframe

  except KeyError:
    print(f"{index_number} не обнаружено")
    return dframe

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def col_add_one(dframe,col_name="any",fillin="nan"):
  if fillin == "nan": fillin = np.NaN
  if col_name == "any":
    dframe[f"{dframe.shape[1]+1}_{col_name}"] = fillin
    return dframe
  else:
    dframe[col_name] = fillin
    return dframe

#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ

def col_rm_one(dframe,col_name="any"):
  """
  :
  dframe_col_rm_one(X_Cols,col_name=-1)
  X_Cols
  :
  """

  try:
    if col_name == "any":
      print("Нужно указать номер \ наименование столбца")
      print("Выполнение завершено.")
      return
    else:
      dframe = dframe.drop(columns=[col_name],inplace=True)
      return dframe
  except (KeyError):
    print(f"{col_name} не обнаружено")
    return dframe
  
#ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ
  
class DFRame():
  """
  :
  empty_dframe = pd.DataFrame()
  :
  dframe_from_list = pd.DataFrame( data_list )
  :
  dframe_from_list = pd.DataFrame( data_list, columns=["column_name_1","column_name_2"] )
  :
  dframe_from_list = pd.DataFrame( data_list, index=["index_name_1","index_name_2"] )
  :
  dframe_from_list = pd.DataFrame( list(zip(data_list, index_list)) )
  :
  dframe_from_list = pd.DataFrame( list(zip(data_list, index_list)), 
  :                                columns=["column_name_1",
  :                                         "column_name_2"] )
  :
  dframe_from_2D_list = pd.DataFrame( [[col_1_val_1,col_2_val_1],
  :                                    [col_1_val_2,col_2_val_2]], 
  :                                    columns=["column_name_1",
  :                                             "column_name_2"] )
  :
  dframe_from_3D_list_vs_dtype = pd.DataFrame( [[col_1_val_1,col_2_val_1,col_3_val_1],
  :                                             [col_1_val_2,col_2_val_2,col_3_val_2],
  :                                             [col_1_val_2,col_2_val_2,col_3_val_3],], 
  :                                             columns=["column_name_1",
  :                                                      "column_name_2",
  :                                                      "column_name_2"], dtype = float )
  :
  dframe_from_dict = pd.DataFrame( {"column_name_1": value} )
  :
  dframe_from_dict = pd.DataFrame( {"column_name_1": [list_values]} )
  :
  dframe_from_dict = pd.DataFrame( {"column_name_1": [list_values],
  :                                 "column_name_2": [list_values]} )
  :
  nd_array_2_dframe = pd.DataFrame( {"column_name_1": nd_array[0],
  :                                  "column_name_2": nd_array[1]} )
  :
  :
  """

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>

  @staticmethod
  def row_add_one(dframe,fillin = "nan"):
    """
    :
    dfr.row_add_one(df_components).fillna("").rename(index={13:""})
    :
    """

    return row_add_one(dframe=dframe,
                       fillin=fillin)

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>

  @staticmethod
  def row_add_any(dframe,count_rows=1, fillin = "nan"):

    return row_add_any(dframe=dframe,
                       count_rows=count_rows,
                       fillin=fillin)

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>

  @staticmethod
  def row_insert_one_before(drame,insert_before_index=1):
    
    return row_insert_one_before(drame=drame,
                                 insert_before_index=insert_before_index)

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
  
  @staticmethod
  def row_insert_one_after(drame,insert_after_index=1):

    return row_insert_one_after(drame=drame,
                                insert_after_index=insert_after_index)

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
  
  @staticmethod
  def row_rm_one(dframe,index_number="all"):
    """
    :
    dframe_row_rm_one(X_Cols,index_number=-1)
    X_Cols
    :
    """

    return row_rm_one(dframe=dframe,
                      index_number=index_number)

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
  
  @staticmethod
  def row_rm_any(dframe,index_number=[]):
    """
    :
    dframe_row_rm_any(X_Cols,index_number=[509,510])
    X_Cols
    :
    """

    return row_rm_any(dframe=dframe,
                      index_number=index_number)  

  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
  
  @staticmethod
  def col_add_one(dframe,col_name="any",fillin="nan"):

    return col_add_one(dframe=dframe,
                       col_name=col_name,
                       fillin=fillin)
  
  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>

  @staticmethod
  def col_rm_one(dframe,col_name="any"):
    """
    :
    dframe_col_rm_one(X_Cols,col_name=-1)
    X_Cols
    :
    """

    return col_rm_one(dframe=dframe,
                      col_name=col_name)
  
  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
  
  @staticmethod
  def row_add_Cols_as_Rows(dframe_out=pd.DataFrame(),out_col_name=[],dframe_in=pd.DataFrame(),orig_name=True):
    
    return row_add_Cols_as_Rows(dframe_out=dframe_out,
                                out_col_name=out_col_name,
                                dframe_in=dframe_in,
                                orig_name=orig_name)
  
  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
  
  @staticmethod
  def col_copy_one_bigger(dframe_in,dframe_out,out_col_name=""):

    return col_copy_one_bigger(dframe_in=dframe_in,
                               dframe_out=dframe_out,
                               out_col_name=out_col_name)
  
  # <:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:><:x:>
