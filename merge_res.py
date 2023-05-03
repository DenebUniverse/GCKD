import csv
import paramiko

# 服务器列表，每个元素包含服务器的IP地址、用户名和密码
server_list = [
    {"ip": "10.0.0.1", "username": "user1", "password": "pass1"},
    {"ip": "10.0.0.2", "username": "user2", "password": "pass2"},
    # 添加更多服务器
]

# 远程csv文件路径，需要根据实际情况修改
remote_csv_path = "/path/to/remote/file.csv"

# 本地csv文件路径，需要根据实际情况修改
local_csv_path = "/path/to/local/file.csv"

# 合并后的csv表头，需要根据实际情况修改
header = ["col1", "col2", "col3", "col4"]

# 创建一个空的列表，用于存储所有服务器的数据
data = []

# 遍历所有服务器
for server in server_list:
    # 创建SSH客户端对象
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 连接服务器
    client.connect(server["ip"], username=server["username"], password=server["password"])

    # 打开SFTP会话
    sftp = client.open_sftp()

    # 下载远程文件到本地
    sftp.get(remote_csv_path, f"{server['ip']}.csv")

    # 关闭SFTP会话和SSH连接
    sftp.close()
    client.close()

    # 读取本地文件内容，并将数据添加到列表中
    with open(f"{server['ip']}.csv", 'r') as file:
        reader = csv.reader(file)
        # 跳过表头
        next(reader)
        for row in reader:
            data.append(row)

    # 删除临时文件
    os.remove(f"{server['ip']}.csv")

# 将所有数据写入本地csv文件中
with open(local_csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(header)
    for row in data:
        writer.writerow(row)
