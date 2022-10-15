from sklearn.preprocessing import QuantileTransformer

def qqTransform(dframe):
  """
  """
  import pandas as pd
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
