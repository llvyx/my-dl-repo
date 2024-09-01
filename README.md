# DL System 仓库 by 赵韶涵

## 仓库文件结构

- `learnnote/`: 总的学习笔记

- `dlnote/`: 学习深度学习过程的笔记

- `task1/`: 任务1的实现代码，其中：

  - `model.py`: 模型训练过程的实现代码，其中包含3个不同架构的CNN网络和2个MLP网络。

  - `test.py`: 模型应用过程的实现代码，从`img_path`中读取拍摄的图片并利用保存的模型做出预测。

- `task2/`: 任务2的实现代码，其中：

  - `3-camera_server/`: 搭建Camera Web Server用到的源代码以及项目文件夹：

    - `project/`: ESP-IDF构建根目录的所有文件（不含`build/`，使用VS Code扩展）

    - `camera.c`: 包含引脚定义以及初始化OV2640摄像头的代码。

    - `main.c`: 包含HTTP服务器的构建代码。

    > 说明：访问`IP地址`后，会出现 "Welcome to ESP32 Web Server" 的文本响应；访问`{IP-address}/capture`后，会收到服务器拍摄的照片响应。

  > 由于`1-hello_world`和`2-blink`主要是在官方示例代码上做很小的修改，故在此处不再贴出

- `task3/`: 任务3的部分结果

- `README.md`: 当前文件
