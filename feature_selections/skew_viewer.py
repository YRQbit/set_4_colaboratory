def feature_skew(dframe):
  
  for k,el in enumerate(dframe.skew()):
    if el >= 1: print(f"смещение вправо _:=:_ \033[34m{dframe.skew().index[k]}\033[0m _:=:_ {dframe.skew()[k].round(3)}")
    if el <= -1: print(f"смещение влево _:=:_ \033[34m{dframe.skew().index[k]}\033[0m _:=:_ {dframe.skew()[k].round(3)}")
    if (1 > el) and (el > -1): print(f"симметричное распределение _:=:_ \033[34m{dframe.skew().index[k]}\033[0m _:=:_ {dframe.skew()[k].round(3)}")
