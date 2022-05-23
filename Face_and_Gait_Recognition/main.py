import numpy as np
import glob
import os
import face
import gait
import similer


def main() :
    FC = face.Faces()
    GT = gait.Gaits()
    # モード選択(登録 or 識別)
    print('Select Mode')
    print('0:Register Mode，1:Detector Mode')

    mode = input()

    if mode == '0':
        print('Selected Register Mode:')
        '''
        登録モード
        フォルダ内の人物をすべて登録．
        img_mov_Path：Data/Original_Data/'UserName'/
        顔画像　：face.jpg
        歩行映像：gait.mp4
        save_Path：Data/User/'UserName'/
        顔特徴　：face.npz
        歩行特徴：gait.npz
        '''
        path = 'Data/Original_Data/*'
        flist = glob.glob(path)
        for f in flist :
            # 人物名(フォルダ名)を取得
            user_name = os.path.basename(f)
            print(user_name)
            img_path = f + '/face.jpg'
            mov_path = f + '/gait.mp4'
            save_path = 'Data/User/' + user_name + '/'

            # ファイルの存在確認
            if not os.path.isfile(img_path) or not os.path.isfile(mov_path):
                print('No face.jpg or gait.mp4')
                continue
            # 顔画像から顔部分の切り出し&特徴取得．
            # In : img_path，Out：face_vector
            face_vector = FC.img2Vector(img_path)

            # 歩行映像からGait Energy Image(GEI)を生成&特徴取得．
            # In : mov_path，Out：gait_vector
            gait_vector = GT.mov2vector(mov_path)

            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            # 顔特徴と歩行特徴を保存．
            np.savez(save_path + 'face', face_vector[0])
            np.savez(save_path + 'gait', gait_vector)

        print('Finished Register Mode')

    elif mode == '1' :
        Sim = similer.Similerity()
        print('Selected Detector Mode')
        '''
        識別モード
        選択された人物が誰かを判定．
        識別対象者の顔画像と歩行映像は以下へ保存しておく必要あり．
        Path = Data/Detector/'UserName'/
        顔画像　：face.jpg
        歩行映像：gait.mp4
        '''
        print('Input Detected User name')
        user_name = input()
        path = 'Data/Detector/' + user_name
        img_path = path + '/face.jpg'
        mov_path = path + '/gait.mp4'
        # ディレクトリの存在確認
        if not os.path.isdir(path):
            print('No user')
            exit()
        # ファイルの存在確認
        if not os.path.isfile(img_path) or not os.path.isfile(mov_path):
            print('No face.jpg or gait.mp4')
            exit()

        # 認証手法選択
        print('Select Authentication Method（0：Face，1：Gait，2：Face + Gait）')
        Method = input()

        if Method == '0':
            # 顔画像から顔部分切り出し & 特徴取得
            # In : img_path，Out：face_vector
            face_vector = FC.img2Vector(img_path)
            # 登録データと類似度計算 & 最も似ている人物特定．
            fname, f_similarity = Sim.compare(face_vector[0], Method)
            print('顔認証　：' + fname + '，' + str(f_similarity))

        elif Method == '1':
            # 歩行映像からGEI生成 & 歩行特徴取得
            # In : mov_path，Out：gait_vector
            gait_vector = GT.mov2vector(mov_path)
            # 登録データと類似度計算 & 最も似ている人物特定．
            gname, g_similarity = Sim.compare(gait_vector, Method)
            print('歩容認証：' + gname + '，' + str(g_similarity))

        elif Method == '2':
            # 顔画像から顔部分切り出し & 特徴取得
            # In : img_path，Out：face_vector
            face_vector = FC.img2Vector(img_path)
            # 歩行映像からGEI生成 & 歩行特徴取得
            # In : mov_path，Out：gait_vector
            gait_vector = GT.mov2vector(mov_path)
            # 登録データと類似度計算 & 最も似ている人物特定．
            mname, m_similarity = Sim.compare_multi(face_vector, gait_vector)

            print('マルチモーダル認証：' + mname + '，' + str(m_similarity))

        else:
            print('No Method.')

    else :
        print('No Mode.')

if __name__ == '__main__':
    main()