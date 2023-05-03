import os
import shutil

# 需要删除的文件夹所在的根目录
root_dir = '/path/to/root/directory/'

# 需要删除的文件夹的条件，例如删除以 "delete_" 开头的文件夹
condition = "delete_"

# 遍历根目录下的所有文件夹
for dirpath, dirnames, filenames in os.walk(root_dir):
    for dirname in dirnames:
        # 判断当前文件夹是否满足条件
        if dirname.startswith(condition):
            # 使用shutil模块删除文件夹
            shutil.rmtree(os.path.join(dirpath, dirname))

# 需要处理的txt文件路径
file_path = '/save/students/tensorboard/tune.txt'

# 需要删除的行所在的条件，例如删除包含 "delete" 关键字的行
condition = "delete"

# 读取文件内容
with open(file_path, 'r') as file:
    lines = file.readlines()

# 遍历所有行，将不满足条件的行保存到新的列表中
new_lines = []
for line in lines:
    if not condition in line:
        new_lines.append(line)

# 将新列表中的行写入原文件中，覆盖原有内容
with open(file_path, 'w') as file:
    file.writelines(new_lines)
