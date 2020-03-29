# PixelFlow
Self supervisedによって学習されたモデル(https://arxiv.org/abs/1909.11895) を利用することで高精度なTrackingを行えるのではという試み。
## 実行方法
```
python PixcelTracker.py
```
## アルゴリズム説明
### 使用モデル
https://github.com/xiaolonw/UVC-1
このモデルはmaskによって指定される枠内のpixelが次のフレーム内のどこに移動したかを予測するモデルである。(いわゆるoptical flow)
これを利用して本コードではtrackingを行なう。
### 入力
データセットMOT17-02の各フレーム画像,初期の人boundingbox,+(2フレームの以降のboundingbox)
2フレーム以降のbboxを使用するかは用途による。使用しなければbboxによる補正がかからず、モデルのみを用いたtrackingになる
### 出力
trackingされたperson(bbox,id)
### 処理の流れ
PixcelTracker.pyが実行コードである。
Tracker.pyが毎フレーム実行されるTrackerのクラスである。
Observer.pyはTracking結果をまとめ、出力するためのクラスである。

まず、Trackerに最初のbboxが与えられると、フレームimage,そのbbox内のpixel,次のフレームimageを使ってbbox内のpixelが次フレームのどこに該当するかを予測する
その後、次フレーム内の予測pixel位置を取り囲むような領域を新たなbboxとして提示する。(このときidは保持されるのでtrackingが可能になる)
もし、次フレーム内のbboxが使用できるなら提示bboxと最大のiouを取るものでmatchingさせる。

現コードはまだ簡易番で最初のbboxのみで新規検出は考慮していない。
occlusionなどがはいりtrackするものがなくなるとtrackingが途切れエラーがでて中断する。
