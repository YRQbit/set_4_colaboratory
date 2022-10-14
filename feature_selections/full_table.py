def print_full(x):
    """
    Назначение: вывод полной ширины ячеек таблицы dframe
    Источник: https://legkovopros.ru/questions/15113/kak-rasshirit-razmer-stolbcza-v-dataframe-python-duplicate
    """
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    display(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

# zs = pvalue_selector(df_pandas)
# print_full(zs)
