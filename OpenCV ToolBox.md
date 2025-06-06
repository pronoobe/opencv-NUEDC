# OpenCV 可视化参数调节工具箱说明

## 功能概述

本工具箱是一个基于 OpenCV 和 Tkinter 的图形化图像处理工具，支持多种常见的计算机视觉算法，并提供直观的参数调节界面。主要功能包括：

1. **图像处理功能**：
   - Canny 边缘检测
   - 图像阈值化（支持多种阈值类型）
   - HSV 颜色空间阈值分割
   - 高斯模糊
   - 膨胀和腐蚀操作
   - 轮廓检测
   - Shi-Tomasi 角点检测
   - 霍夫直线检测（概率方法）
   - 霍夫圆检测

2. **交互功能**：
   - 上传本地图像或通过摄像头实时捕获
   - 直观的参数调节滑块和控件
   - 实时预览处理效果
   - 保存处理后的图像
   - 保存和加载参数配置

3. **代码生成**：
   - 自动生成对应的 Python/OpenCV 代码
   - 一键复制代码到剪贴板

## 使用说明

1. **图像输入**：
   - 点击"上传图像"按钮选择本地图像文件
   - 或点击"启动摄像头"使用摄像头实时捕获

2. **功能选择**：
   - 从下拉菜单中选择需要的图像处理功能
   - 界面会自动显示该功能对应的可调参数

3. **参数调节**：
   - 使用滑块、下拉菜单等控件调节参数
   - 处理结果会实时更新显示

4. **结果保存**：
   - 点击"保存处理结果"保存处理后的图像
   - 点击"保存参数"将当前参数设置保存为 JSON 文件

5. **代码生成**：
   - 底部文本框会自动生成对应的 OpenCV 代码
   - 点击"复制代码到剪贴板"可复制代码

## 技术特点

1. **模块化设计**：
   - 每个处理功能都是独立的 Python 函数
   - 易于扩展新的处理算法

2. **用户友好界面**：
   - 参数控件根据参数类型自动生成
   - 实时反馈处理结果
   - 错误处理和提示信息

3. **跨平台支持**：
   - 基于 Python 和 Tkinter，支持 Windows/Linux/macOS

## 适用场景

- OpenCV 学习和教学演示
- 计算机视觉算法参数调试
- 快速原型开发和验证
- 图像处理工作流程中的参数优化

## 系统要求

- Python 3.x
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- Tkinter (通常随 Python 一起安装)
- pyperclip(用于把生成的代码自动复制到剪切板)
这个工具箱特别适合需要快速验证 OpenCV 算法效果或调试参数的场景，通过可视化交互大大提高了工作效率。
