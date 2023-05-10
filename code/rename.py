import os

# 获取当前文件夹下所有图片的文件名
files = [f for f in os.listdir('./img/left') if os.path.isfile(f) and f.lower().endswith('.jpg')]

# 排序文件名列表
files.sort()
if files == []:
    print('No files found.')
    exit()
# 遍历文件名列表，重命名文件
for i, f in enumerate(files):
    ext = os.path.splitext(f)[1]
    new_name = '{}{}'.format(i+1, ext)
    os.rename(f, new_name)
    print('Renamed {} to {}'.format(f, new_name))
