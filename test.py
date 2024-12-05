import os

# 打印 Mac 上的常用字体文件
font_dir = "/System/Library/Fonts/Supplemental"
fonts = [f for f in os.listdir(font_dir) if f.endswith('.ttc')]
print("系统字体文件:", fonts)
