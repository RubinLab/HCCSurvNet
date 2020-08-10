import pandas as pd

def selectX(df_dict=df_dict, ids=ids, x=100):
    selectX = pd.DataFrame(columns=df_dict[ids[0]].columns)
    for i in ids:
        selectX = selectX.append(df_dict[i].sort_values(by=['prob'], ascending=False)[:x])
    selectX.prob.plot.box()
    plt.show()
    selectX.prob.plot.hist(bins=40)
    plt.show()
    pd.DataFrame(selectX.fname).to_csv('/path/to/save/csv', index=False)

if __name__ == '__main__':
    path = '/path/to/csv'
    df = pd.read_csv(path)

    df_dict = {}
    for name, group in df.groupby('id'):
        df_dict[name] = group
    ids = list(df_dict.keys())

    selectX(df_dict, ids, x=100)