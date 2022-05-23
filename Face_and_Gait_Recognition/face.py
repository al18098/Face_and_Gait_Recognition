import cv2
import dlib
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import numpy as np

# 顔部分の切り出し(dlib)
class Faces:
    def __init__(self):
        # dlib呼び出し
        self.predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(self.predictor_path)
        # ResNet-50モデル呼び出し
        self.model = VGGFace(include_top=False, model='resnet50',
                             pooling='max')  # default : VGG16 , you can use model='resnet50' or 'senet50'

    # img_pathから画像呼び出し & (img_size, img_size)へサイズ正規化
    def Extract(self, img_path, img_size=224):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        faces = self.detector(img, 1)
        if len(faces) < 1 :
            print('Cannot find face.')
            return False, img

        # 画像から見つけた顔位置から切り出し
        face = faces[0]
        top = face.top()
        bottom = face.bottom()
        left = face.left()
        right = face.right()

        # 顔の中心位置をとる
        H = bottom - top
        W = left - right
        cx = int(left - (W/2))
        cy = int(bottom - (H/2))

        # 顔を正方形で切り出す(縦横の長いほうに長さを合わせる)
        if H >= W :
            length = H
        else :
            length = W

        # 切り出し座標を設定(左上の座標と右下の座標)
        tlx = cx - int(1.5 * (length/2))
        tly = cy - int(1.5 * (length/2))
        brx = cx + int(1.5 * (length/2))
        bry = cy + int(1.5 * (length/2))
        # 切り出し位置が画面の外にある場合はダメ
        if tlx < 0 or tly < 0 or brx > width or bry > height :
            print('Face is outside of image area.')
            return False, img

        ex_img = img[tly: bry, tlx:brx]
        ex_img = cv2.resize(ex_img, (img_size, img_size))

        return True, ex_img

    # 顔特徴取得
    def Feature(self, img):
        # imgはRGBでとってくる必要があるのでBGR -> RGB変換
        # img.shape = (224, 224, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)  # if VGG16 : version=1. elif resnet50 or senet50 : version=2.
        # 特徴量 (1, 2048)次元
        preds = self.model.predict(x)
        return preds

    # 顔画像のパスから特徴ベクトル取得
    def img2Vector(self, img_path, img_size=224):
        status, img = self.Extract(img_path, img_size)
        # 上手く顔が見つけられなけらば強制終了.
        if not status :
            print('Interruptions.')
            exit()
        preds = self.Feature(img)
        return preds


