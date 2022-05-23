
import numpy as np
import glob
import os
# 類似度計算　& 最も似ている人物探索

def _load_pca_vals(vecs_path):
    # 特徴ベクトルを呼び出す
    npz1 = np.load(vecs_path)
    vector = npz1['arr_0']
    return vector

class Similerity:
    # 登録データと入力データの比較(style:顔・歩容のどれか)
    def compare(self, d_vector, style):
        max_cos = -1
        max_name = ''

        if style == '0':
            # 顔認証
            flist = glob.glob('Data/User/*')
            for f in flist:
                name = os.path.basename(f)
                r_vector = _load_pca_vals(f + '/face.npz')
                cos = self.cos_dif(d_vector, r_vector)

                if cos > max_cos:
                    max_name = name
                    max_cos = cos

        elif style == '1':
            # 歩容認証
            flist = glob.glob('Data/User/*')
            for f in flist:
                name = os.path.basename(f)
                r_vector = _load_pca_vals(f + '/gait.npz')
                cos = self.cos_dif(d_vector, r_vector)

                if cos > max_cos:
                    max_name = name
                    max_cos = cos

        return max_name, max_cos

    def compare_multi(self, fd_vector, gd_vector):
        max_cos = -1
        max_name = ''
        # 重み
        fw = 0.5
        gw = 0.5
        sw = 0

        flist = glob.glob('Data/User/*')
        for f in flist:
            name = os.path.basename(f)
            fr_vector = _load_pca_vals(f + '/face.npz')
            gr_vector = _load_pca_vals(f + '/gait.npz')
            fcos = self.cos_dif(fd_vector, fr_vector)
            gcos = self.cos_dif(gd_vector, gr_vector)
            # 重み付き和
            mcos = fw * fcos + gw * gcos + sw
            if mcos > max_cos:
                max_name = name
                max_cos = mcos

        return max_name, max_cos[0]




    # コサイン距離の計算
    def cos_dif(self, d_vector, r_vector):
        abf = np.dot(d_vector, r_vector) / (np.linalg.norm(d_vector) * np.linalg.norm(r_vector))
        return abf