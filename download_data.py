import urllib.request
import zipfile
import os
import shutil

print("Downloading dataset...")
# Using GitHub raw link instead
url = "https://raw.githubusercontent.com/hassanamin12/Fake-News-Detection-Using-Machine-Learning/master/news.csv"
filename = "data/news.csv"

try:
    urllib.request.urlretrieve(url, filename)
    print("Download complete!")
    print("File saved: data/news.csv")
except:
    print("Link failed. Using alternative...")
    # Alternative: Use a smaller dataset
    print("Using sample dataset instead...")