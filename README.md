# Google Colaboratory適合Python初學者的雲端開發環境
南華大學跨領域-人工智慧期中報告 11218122 李冠霖
# Colaboratory簡介

此文章在介紹Google Colaboratory的開發環境與解釋下Hello World等級的機器學習操作過程。

推廣colab並表達這網站對入手程式碼的方便與入門操作。

# Colab Notebook環境介紹
開啟chrome先登入google帳號

進入colab連結URL https://colab.research.google.com/ 

出現對話窗如下

![image](https://github.com/linggon-c0der/Google-Colaboratory-Python-/blob/main/202003200701.jpg)

按下右下角 NEW PYTHON 3 NOTEBOOK，出現如下的cell code區域。

![image](https://github.com/linggon-c0der/Google-Colaboratory-Python-/blob/main/202003200701.jpg)

點擊 code cell進入編輯模式並貼上這段python程式碼：

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

按下左側執行button 或CTRL+Enter。會看到正態分布直方圖結果如下：
