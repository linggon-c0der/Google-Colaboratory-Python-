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

![image](https://github.com/linggon-c0der/Google-Colaboratory-Python-/commit/57f8b52052edffe54c6883b703e38ae7f46bfd88)

Colab有code IntelliSense功能，以上述範例來說，在前面兩行import完numpy等函式庫後，請先按下執行。接著再寫 x= numpy.random.n…編輯器會自動顯示代碼完成，參數信息，快速信息和成員列表等功能

如圖:
![image]

滑鼠移至code cell右上角RAM/Disk區域時，會顯示本次執行虛擬機所分配的資源：約12GB RAM，100GB Disk，如下圖。
![image]

# 版本比較Revision History
點選工具列File -> Revision History，或直接於主工具列上按下All changes saved，有時會顯示Last saved at…(某時間點)。你會看到幾乎任何時間點的更動都被記錄下來，可以做檔案比較、選擇再次開啟歷史版本或復原版本。

![image]

# 於notebook (副檔名.ipynb)中任意加入code cell或text cell
由於Colab是基於Jupyter Notebook發展出來的互動式環境，可以任意加入code cell(使用Python和其他語言編寫和執行代碼)或text cell(使用稱為markdown的簡單標記語言進行格式化筆記)。在下圖中，「下面的global變數x值是由上面這個cell得來，且不用再import函式庫，即可畫出Scatter Plot。」這段敘述是在Text cell中，同樣用點擊區塊方式進入編輯模式。新增這些cells方式也很簡單，將滑鼠移到cell的上方或下方皆可。

![image]

接下來編輯執行第二個cell code，程式碼如下：

y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()

得到錯誤訊息如下圖：

![image]

原來是x和y size必須相同。這是很容易修正的錯誤訊息，但是若一時看不出來問題在那兒呢？Colab在ValueError:下提供一個按鈕

![image]，方便我們查詢網站stack overflow的相關解答，非常實用。修改後就可畫出正確分布圖形了：

![image]

每一個code cell右上角都具有簡易工具列如下圖，最右邊的More cell actions中有Clear output功能，可快速清理output視窗。

![image]

# Mounting Google Drive in your VM

展開Colab notebook左邊的區域，google提供很多方便的Code snippets範例程式碼，我們挑選存取google雲端硬碟的範例實作。

![image]

首先要mount上google drive，程式碼如下。
這裡的設定需要綁定權限，請按照指示，連上google oauth2 URL後認證，並複製貼上你的authorization code。

![image]

from google.colab import drive
drive.mount('/gdrive')

當輸入驗證完成，會顯示Mounted at /gdrive，這就表示成功了。

![image]

接著在”我的雲端硬碟”中新增檔案foo.txt，並列印出內容。程式碼及執行結果如下：

with open('/gdrive/My Drive/foo.txt', 'w') as f:
  f.write('您好 Google Drive!')
!cat '/gdrive/My Drive/foo.txt'

![image]

但我們其實應該到綁定google帳號的雲端硬碟去檢查，檔案是否真的寫入內容了。

![image]

針對google drive的存取，也可以利用python的PyDrive函式庫簡化對Google Drive API的使用，相關範例如下：

```
# Import PyDrive and associated libraries.
# This only needs to be done once in a notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Create & upload a text file.
uploaded = drive.CreateFile({'title': 'PyDriveSample.txt'})
uploaded.SetContentString('Sample upload file content 範例')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))
```
![image]

因為會使用Google Cloud SDK，故執行時也會需要輸入驗證碼，此範例會傳回file ID供後續使用。
接下來測試列出.txt檔案，因為在同一本notebook，上面的函式庫及參數可以直接應用：

# List .txt files in the root.
# Search query reference:
# https://developers.google.com/drive/v2/web/search-parameters
listed = drive.ListFile({'q': "title contains '.txt' and 'root' in parents"}).GetList()
for file in listed:
  print('title {}, date {}, id {}'.format(file['title'], file['createdDate'], file['id']))

  接下來測試下載特定檔案：

# Download a file based on its file ID.
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
file_id = '填入你自己的file ID'
downloaded = drive.CreateFile({'id': file_id})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))

![image]

以上這些檔案操作練習讓我們了解如何使用google drive，之後我們便可將數據資料上傳，以供機器學習使用。
以上這些檔案操作練習讓我們了解如何使用google drive，之後我們便可將數據資料上傳，以供機器學習使用。
