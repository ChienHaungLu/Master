import gdown
import os
# a file
url = "https://drive.google.com/drive/folders/1WzpETjC1qfGCXA1JmoGHDv5OaiQX3pcZ"
# output 指的是您想要儲存的本地資料夾名稱，若未指定會自動使用雲端的資料夾名稱
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "..", "raw")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
# 下載整個資料夾
gdown.download_folder(url, output=OUTPUT_FOLDER, quiet=False, use_cookies=False)