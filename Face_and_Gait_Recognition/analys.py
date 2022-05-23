import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import glob
import cv2

# 最初に実行する前に主成分分析を行う(analys.pyを実行)
# CASIA B datasetよりGEIのデータをダウンロードして，Data/PCA/へ置く．

# 固有ベクトルを計算(主成分を見つける)
def eigen_vecotr():
    # 固有ベクトルのリスト
    e_vec = []
    # GEI格納用のリスト
    g_list = []
    # GEIを読みだして一次元化＆リストへ格納．
    path = 'Data/PCA/gei/*/*/*-090.png'
    flist = glob.glob(path)
    for f in flist:
        img = cv2.imread(f, flags=-1)
        img_flat = img.flatten()
        g_list.append(img_flat)

    # g_listをnumpy.arrayへ変換&平均をとる
    g_list = np.array(g_list)
    g_mean = np.mean(g_list, axis=0)
    # すべてのGEIと平均の差をとる
    for i in range(len(g_list)):
        g_list[i] = g_list[i] - g_mean
    # ndarraylist -> dataframe 変換
    g_list_pd = pd.DataFrame(g_list.T)
    # 主成分分析
    pca = PCA(n_components=len(g_list))
    pca.fit_transform(np.transpose(g_list_pd))
    # 固有ベクトルの算出
    v = pca.components_
    # 寄与率の計算
    rate = pca.explained_variance_ratio_
    # 利用する次元数を計算
    num = contribution_rate(rate)
    # 次元圧縮(必要な次元の固有ベクトルのみをe_vecへ格納)
    for i in range(num):
        e_vec.append(v[i])

    # list -> numpy.array 変換
    e_vec = np.array(e_vec)

    return g_mean, e_vec


# 寄与率が95%を超えるときの固有ベクトルの次元数を返す．
def contribution_rate(rate):
    rat_sum = 0
    num = 0
    while (rat_sum < 0.95):
        rat_sum += rate[num]
        num += 1
    return num

if __name__ == '__main__':
    mean, e_vector = eigen_vecotr()

    np.savez('Data/PCA/pca_mean', mean)
    np.savez('Data/PCA/pca_vec', e_vector)