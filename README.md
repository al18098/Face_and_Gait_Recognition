# Face_and_Gait_Recognition  
**システム内容**
  
![syoumeisyashin_man](https://user-images.githubusercontent.com/66660848/170674778-0325e44a-5cd0-4d73-b5af-d44b89097f17.png)
![画像3](https://user-images.githubusercontent.com/66660848/170675533-7eac2c08-95fa-40fc-9a3e-28498643b558.png)


顔認証と歩容認証、顔と歩容を組み合わせたマルチモーダル認証を行うシステム。　　
  
顔認証では、顔の映っている画像から顔位置検出のためのライブラリであるdlibを利用して顔を切り出し、サイズの正規化(224×224)を行う。  
切り出した顔写真を深層学習のアーキテクチャでるResNet-50(データセットVGGFaceで学習済)へ入力して、顔特徴ベクトルを取得。  
登録データの特徴ベクトルとのcos類似度を算出して，類似度が最も大きい人物であると判断する。
  
歩容認証では、まず歩行者を横から撮影した映像から1フレーム毎に歩行者のシルエットを抽出(クロマキー処理)する。  
その後、歩行者部分のみを切り出して、サイズの正規化(240×240)を行う。  
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
    
**実験**  
  
顔認証と歩容認証を組み合わせることにより、マスク着用時の認証精度が向上するという仮説の元、
マスク非着用時の顔認証・歩容認証・マルチモーダル認証と、マスク着用時の顔認証・歩容認証・マルチモーダル認証の認証精度を測定して、
マスクを着用することによる認証精度への影響と、顔認証と歩容認証を組み合わせることによる認証精度への影響を調査する。  
  
・実験データ：  
   10名の実験参加者に対して顔写真と歩行映像を、マスク非着用の状態で3セット、マスク着用の状態で2セットそれぞれ撮影を行った。  
   そのうち、登録用データとしてマスク非着用の状態の顔写真と歩行映像を1セット利用して、残りのマスク非着用状態・マスク着用状態の2セットずつを識別(検証)用データとして利用する。  
  
  ![datas](https://user-images.githubusercontent.com/66660848/170690313-89b093e7-101e-474f-a276-c953474ee3cb.png)  
  
・評価指標：  
  評価指標として、類似度が最も大きい人物が正解である割合である1位認証率を利用する。  
  1位認証率は以下のように計算を行う。  
  Rank1(%) = (R_p / (R_p + R_o)) × 100  
  R_pは類似度が最も大きい人物が正解の場合のケース数、R_oは不正解の場合のケース数である。  
    
・結果：  
 結果を下の表に示す。  
 各認証手法において、マスクの着用による認証精度に対する影響を見ると、  
 顔認証では、マスクの着用により1位認証率が低下した。  
 歩容認証では、マスク着用時の1位認証率がマスク非着用時の1位認証率を上回った。  
 マルチモーダル認証では、マスクの着用による1位認証率の変化はなかった。  
   
 一方、顔認証と歩容認証を組み合わせることによる認証精度に対する影響を見ると、  
 マスク非着用時の1位認証率は、顔認証とは変わらず100%であり、歩容認証よりも高くなっている。  
 マスク着用時の1位認証率は、顔認証よりも高くなり、歩容認証とは変わらず100%である。
  
  
  ![203334](https://user-images.githubusercontent.com/66660848/170691737-ce59f002-4e48-4581-944b-e76e96ae50f4.png)
  
・考察
　以上の結果から、  
 顔認証では、マスクを着用することにより、認証精度が低下すると考えられる。  
 歩容認証では、マスクを着用することによる認証精度への影響は少ないと考えられるが、登録データと識別データの間の歩行姿勢の違いが認証精度に影響を与えていると考えられる。  
 マルチモーダル認証では、顔認証におけるマスクの着用による影響と歩容認証における歩行姿勢による影響を吸収して、認証精度を向上させることができると考えられる。  
   
   
 このことから、顔認証と歩容認証を組み合わせることにより、マスク着用時の認証精度が向上するという仮説は正しいと考えられる。
 しかし、本実験は少数の実験参加者に対して、制御された環境下で撮影されたデータを実験に利用したため、全体的な精度がかなり高くなり、精度の変化が分かり難くなってしまった。  
 そのため、今後はより大規模なデータを利用した検証が必要であると考える。  
 



  
