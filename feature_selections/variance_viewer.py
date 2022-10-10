def dframe_variance(dframe):
  
  for el in dframe.columns:
    print(f"\033[34m{el}\033[0m",dframe.var()[el].round(3))
