import os, glob
import pandas as pd

def selectX(df_dict=df_dict, ids=ids, x=100):
    selectX = pd.DataFrame(columns=df_dict[ids[0]].columns)
    for i in ids:
        selectX = selectX.append(df_dict[i].sort_values(by=['prob'], ascending=False)[:x])
    return pd.DataFrame(selectX.fname)

def savetopX(svs_tiles=svs_tiles, df=df, savepath=savepath, x=100):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    lis = []
    for f in svs_tiles:
        if os.path.basename(f) in df.fname.tolist():
            lis.append(f)
    for l in lis:
        src = l
        dst = savepath+os.path.basename(l)
        copyfile(src, dst)

if __name__ == '__main__':
    path = '/path/to/tumor_tile_inference_result.csv'
    df = pd.read_csv(path)

    df_dict = {}
    for name, group in df.groupby('id'):
        df_dict[name] = group
    ids = list(df_dict.keys())

    df = selectX(df_dict, ids, x=100)

    svs_tiles = glob.glob('/path/to/svs_tiles/*/*.png')
    savepath = '/path/to/save/top100tiles/'
    savetopX(svs_tiles, df, savepath, x=100)