import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

warnings.simplefilter(action="ignore", category=FutureWarning)

def boxplots_viewer(dframe,figsizer=(13,8)):
  count_columns = len(dframe.columns)
  lst_columns = dframe.columns

  fig, axes = plt.subplots(figsize = figsizer,nrows=count_columns, ncols=2)

  index=0
  for row in axes:
    k=0
    for col in row:
      # print("k ::",k,"col ::",col)
      if k==0:
        # print(lst_columns,lst_columns[index])
        sns.boxplot(data = dframe, # dframe[ lst_columns[index] ],
                    x = lst_columns[index], # x = dframe[lst_columns[index]], 
                    # palette='colorblind',
                    color='#CCCCCC',
                    showfliers=False,
                    showmeans=True,
                    ax=col)
      elif index <= (count_columns-1):
        # print(lst_columns,lst_columns[index])
        sns.boxplot(data = dframe, # dframe[ lst_columns[index] ],
                    x = lst_columns[index],
                    # y = "price_doc",
                    color='#CCCCCC',
                    showfliers=True,
                    showmeans=True,
                    ax=col);
      k+=1
    index+=1


def pairplot_viewer(dframe):
  """
  """
  sns.pairplot(dframe)
  
  
def qqplot_viewer(dframe):
  """
  """
  for column in dframe.columns:
    qqplot(dframe[column], line="s", color="#CCCC99")
  pyplot.show


def sns_heatmap(dframe):
  sns.heatmap(dframe.corr(),
              xticklabels=dframe.corr().columns, 
              yticklabels=dframe.corr().columns, 
              cmap='RdYlGn', 
              center=0, 
              annot=True)
