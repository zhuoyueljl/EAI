# cv_test OpenCV 示例

这个示例基于 Conda 环境 `cv_test`，演示了：

- 使用 OpenCV 读取图片
- 进行灰度化、模糊、边缘检测
- 保存处理后的图片
- 生成一张包含测速信息的可视化对比图

## 运行

在 VS Code 里打开当前项目后，默认解释器已经切到：

`C:\Users\Lijunliang\miniconda3\envs\cv_test\python.exe`

也可以在终端里运行：

```powershell
conda activate cv_test
python opencv_cv_test.py
```

## 输出

脚本会自动创建以下目录和文件：

- `assets/input_demo.png`: 如果没有输入图，会自动生成一张样例图
- `outputs/opencv_processed.png`: OpenCV 处理后并叠加测速文本的图片
- `outputs/opencv_visualization.png`: Matplotlib 拼接出来的可视化对比图

## 在 VS Code 中使用

- 直接打开 `opencv_cv_test.py`
- 点击右上角运行按钮即可
- 如果你使用 Notebook，也可以选择内核 `Python (conda: cv_test)`
