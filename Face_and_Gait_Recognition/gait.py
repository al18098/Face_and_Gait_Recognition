import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import argrelmin, argrelmax


def _load_pca_vals(vecs_path, mean_path):
    # 特徴ベクトルを呼び出す
    npz1 = np.load(vecs_path)
    pca_vec = npz1['arr_0']
    # データセットの画像の平均を呼び出す
    npz2 = np.load(mean_path)
    pca_mean = npz2['arr_0']
    return pca_vec, pca_mean


# 歩行映像からフレーム毎にシルエット抽出．
# GEIを生成．
# 歩行特徴取得
# class1 : 映像のパスからGEIを生成　→ 特徴ベクトル取得
class Gaits:
    def __init__(self):
        self.SE = Silhouette()
        self.pca_vec, self.pca_mean = _load_pca_vals('Data/PCA/pca_vec.npz', 'Data/PCA/pca_mean.npz')

    def gei(self, mov_path):
        img_list = []
        cap = cv2.VideoCapture(mov_path)
        ret, bg = cap.read()
        if not ret:
            print('no videos')
            return 0

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                #print('no frame')
                break
            # フレームから正規化シルエット取得
            img = self.SE.norm_silhouette(frame)
            if len(img) <= 0:
                continue
            # シルエットを時系列順に並べる
            img_list.append(img)
            # 以下, debug用
            cv2.imwrite('Data/debug/Gait/' + str(len(img_list)) + '.jpg', img)
        GEI = self.SE.get_gei(img_list)
        cv2.imwrite('Data/debug/Gait/GEI.jpg', GEI)
        return GEI

        # Gait Energy Imageを生成

    def Feature(self, img):
        img_flat = img.flatten()
        phi = img_flat - self.pca_mean
        # w = v ・ phiより
        preds = np.dot(self.pca_vec, phi)
        return preds

    def mov2vector(self, mov_path):
        GEI = self.gei(mov_path)
        preds = self.Feature(GEI)
        return preds

# class2 : 映像からシルエットを抽出 → 歩行周期取得　→ GEI生成して返す．
class Silhouette:
    def __init__(self, img_size=240):
        self.img_size = img_size
        self.kernel = np.ones((5, 5), np.uint8)

    # フレームからシルエットを切り出し，サイズを正規化して返す．
    def norm_silhouette(self, img):
        height, width = img.shape[:2]
        # (キーイング)マスク画像取得
        mask = self.ex_silhouette(img)
        # 端を黒に塗りつぶし．(使っている映像の端がグリーンバクでないため邪魔になる)
        mask2 = cv2.rectangle(mask, (0, 0), (int(width), int(height / 5) + 40), (0, 0, 0), -1)
        mask2 = cv2.rectangle(mask2, (0, 0), (int(width / 5) + 30, int(height)), (0, 0, 0), -1)
        mask2 = cv2.rectangle(mask2, (int((3 * width) / 4) - 40, 0), (int(width), int(height)), (0, 0, 0), -1)

        # 最大部分のみ残す(人のシルエット部分のみを残す)
        s_img = self.max_area(mask2)

        # 輪郭抽出(シルエットの位置を探索)
        contours, hierarchy = cv2.findContours(
            s_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 小さい輪郭は誤検出として削除
        contours = list(filter(lambda x: cv2.contourArea(x) > 10000, contours))
        if len(contours) <= 0:
            return []
        # 輪郭の外接矩形の位置
        x, y, w, h = cv2.boundingRect(contours[0])
        if int(w - h) >= int(width / 2):
            return []
        # Boxの中心座標
        MY, MX = int((2 * y + h) / 2), int((2 * x + w) / 2)
        # Boxの半径
        R = (y + h) - MY
        s_img = s_img[y:y + h, MX - R:MX + R]
        s_img = cv2.resize(s_img, (self.img_size, self.img_size))
        s_img = self.horizontal_centering(s_img)
        return s_img

    # 画像を2値化する
    def ex_silhouette(self, img):
        # HSV色空間に変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 緑色のHAVの値域１
        hsv_min = np.array([30, 64, 0])
        hsv_max = np.array([90, 255, 255])
        # 緑色領域のマスク
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        # オープニング処理
        erosion = cv2.erode(mask, self.kernel, iterations=3)
        deliation = cv2.dilate(erosion, self.kernel, iterations=3)
        # マスキング処理
        masked_img = cv2.bitwise_and(img, img, mask=deliation)
        # マスク
        hsv2 = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv2, hsv_min, hsv_max)
        # 白黒反転
        mask2 = cv2.bitwise_not(mask2)

        return mask2

    # 2値画像の中で最も大きい面積の前景のみを残し，あとは背景とする．
    def max_area(self, img):
        contours = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # 一番面積が大きい輪郭を選択する。
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # 黒い画像に一番大きい輪郭だけ塗りつぶして描画する。
        out = np.zeros_like(img)
        cv2.drawContours(out, [max_cnt], -1, color=255, thickness=-1)
        return out

    # シルエット上半分から水平方向にセンタリングを行う(シルエットの位置合わせ)
    def horizontal_centering(self, img):
        height, width = img.shape[:2]
        img_cut = img[0:int(height / 3), 0:width]

        contours, hierarchy = cv2.findContours(img_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            mid_x = int((x + (x + w)) / 2)
            hor_warp = int((width / 2) - mid_x)
            M = np.float32([[1, 0, hor_warp], [0, 1, 0]])
            # , borderValue=255
            img_center = cv2.warpAffine(img, M, (width, height))
        else:
            img_center = img
        return img_center

    # シルエットシーケンスからGEIを生成スル．
    def get_gei(self, img_list):
        h, w = img_list[0].shape[:2]
        # GEI計算のための宣言(平均画像を作るため足しこんでいく)
        GEI = np.zeros((h, w))
        # 歩行周期を取得．(足を最も開いている時のフレーム番号を取得)
        maxid, minid = self.get_cycles(img_list, h, w)
        # 1歩行周期分の平均画像を計算
        if len(maxid[0]) < 3:
            print('Not enough walking time.')
            exit()

        for i in range(maxid[0][0], maxid[0][2]):
            GEI += img_list[i]

        GEI = GEI / (maxid[0][2] - maxid[0][0])
        return GEI

    # シルエットシーケンスから歩行周期を求める
    def get_cycles(self, img_list, h, w):
        # 下半身の前景ピクセル数をカウントする
        f_sum_list = [] # 前景ピクセル数リスト
        label_list = []  # 画像ラベルリスト
        for i in range(len(img_list)):
            img = img_list[i]
            img_foot = img[int(h/2):h, 0:w]
            # ndarray変換
            nd_img = np.array(img_foot)
            f_sum = np.count_nonzero(nd_img == 255)
            f_sum_list.append(f_sum)
            label_list.append(i)

        X = np.array(label_list)
        Y = np.array(f_sum_list)
        maxid = argrelmax(Y, order=7)
        minid = argrelmin(Y, order=5)
        # 以下 歩行周期debug
        fig = plt.figure()
        plt.plot(X, Y, '-k', label='original')
        plt.plot(X[maxid], Y[maxid], 'ro', label='Peak (max)')
        plt.plot(X[minid], Y[minid], 'bo', label='Peak (min)')
        plt.legend()
        fig.savefig('Data/debug/Gait/Cycle.png')
        return maxid, minid

