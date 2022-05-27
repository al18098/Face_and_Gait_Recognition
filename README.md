# Face_and_Gait_Recognition  
**システム内容**
  
![syoumeisyashin_man](https://user-images.githubusercontent.com/66660848/170674778-0325e44a-5cd0-4d73-b5af-d44b89097f17.png)
![画像3](https://user-images.githubusercontent.com/66660848/170675533-7eac2c08-95fa-40fc-9a3e-28498643b558.png)


顔認証と歩容認証、顔と歩容を組み合わせたマルチモーダル認証を行うシステム。　　
  
顔認証では、顔の映っている画像から顔位置検出のためのライブラリであるdlibを利用して顔を切り出す。  
切り出した顔写真を深層学習のアーキテクチャでるResNet-50(データセットVGGFaceで学習済)へ入力して、顔特徴ベクトルを取得。  
登録データの特徴ベクトルとのcos類似度を算出して，類似度が最も大きい人物であると判断する。
  
歩容認証では、まず歩行者を横から撮影した映像から1フレーム毎に歩行者のシルエットを抽出(クロマキー処理)する。  
その後、歩行者部分のみを切り出して、サイズの正規化を行う。  
サイズを正規化したシルエット画像列から、下半身の前景ピクセル数をカウントして左右2歩を踏み出す時間である1歩行周期を算出。  
頭の位置を揃えた1歩行周期分の歩行者シルエット画像列から平均画像であるGait Energy Image(GEI）を算出。  
GEIから主成分分析により、歩行特徴ベクトルを算出。
登録データの特徴ベクトルとのcos類似度を算出して，類似度が最も大きい人物であると判断する。
  
マルチモーダル認証では、顔と歩容からそれぞれ算出したcos類似度を重み付き和で統合する。  
統合後の類似度から、最も似ている人物を判断する。  
  
  ![System](https://user-images.githubusercontent.com/66660848/170673537-20e0dc2d-a96d-4dec-a078-4e33bcd05554.jpg)　　
  
**利用方法**  
用意するもの：  
　・顔の映っている画像  <- face.jpg
　・背景がグリーンバックの歩行者を横から撮影した映像 <- gait.mp4

実行前に：  
　1.CASIA Gait Database BからGEIのデータセットをダウンロードして、'Data/PCA/'に置き、analys.pyを実行  
　2.'Data/Original_Data/UserName/'へ登録するための顔の映っている画像(face.jpg)と歩行映像(gait.mp4)を置く  
　3.'Data/Detector/UserName/'へ識別するための顔の映っている画像(face.jpg)と歩行映像(gait.mp4)を置く  
   
 実行：  
   ・main.pyを実行する  
     
     Select Mode　＝　登録モード(0)と識別モード(1)のどちらかを選択。
     0:Register Mode，1:Detector Mode
      -> 0 or 1
     
     ・登録モード(0)のとき
     　'Data/Original_Data/'内の動画像から自動でユーザ登録。
      
     ・識別モード(1)のとき
     　Input Detected User name　＝ 'Data/Detector'内にあるフォルダ名(ユーザ名)を入力。
        -> User_Name
       
       Select Authentication Method（0：Face，1：Gait，2：Face + Gait）＝ 顔認証(0)・歩容認証(1)・マルチモーダル認証(2)から認証手法を選択。
        -> 0 or 1 or 2
        
   ・「認証手法：登録データの中で最も似ている人物名，最大類似度」が出力される。
    
実験：  
  
  
