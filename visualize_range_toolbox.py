# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import inspect  # 用于获取函数参数信息
import json  # For saving/loading parameters
import pyperclip  # For copying code to clipboard
import os  # For saving files
import time  # For potential delays or FPS limiting

# --- 全局变量和常量 ---
MAX_IMAGE_WIDTH = 400  # 图像显示最大宽度
MAX_IMAGE_HEIGHT = 400  # 图像显示最大高度
CAMERA_UPDATE_DELAY_MS = 30  # 大约 33 FPS, 根据需要调整


# --- OpenCV 函数处理模块 ---
# (在此处保留问题中提供的所有 process_image_* 函数)
# --- START: Functions from the prompt ---
def process_image_hsv_threshold(img, params):
    """ Applies color thresholding in HSV space """
    if len(img.shape) != 3 or img.shape[2] != 3:
        # Need a color image for HSV conversion
        # Create a black image with text
        h, w = (200, 400) if len(img.shape) < 2 else img.shape[:2]
        error_img = np.zeros((h, w), dtype=np.uint8)
        # 注意：错误信息文本保持英文，以便调试
        return cv2.putText(error_img, "Input must be a color image", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_lower = params.get('H_Lower', 0)
    s_lower = params.get('S_Lower', 50)
    v_lower = params.get('V_Lower', 50)
    h_upper = params.get('H_Upper', 179)
    s_upper = params.get('S_Upper', 255)
    v_upper = params.get('V_Upper', 255)

    lower_bound = np.array([h_lower, s_lower, v_lower], dtype=np.uint8)
    upper_bound = np.array([h_upper, s_upper, v_upper], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask  # Return the binary mask


def process_image_hough_circles(img, params):
    """ Detects circles using Hough Circle Transform """
    if len(img.shape) == 3:
        # HoughCircles 通常在灰度图上效果更好
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 应用高斯模糊可以减少噪声，提高检测稳定性
        blur_ksize = params.get('Blur_Kernel_Size', 5)
        # 确保核大小为奇数
        blur_ksize = int(blur_ksize) if int(blur_ksize) % 2 != 0 else int(blur_ksize) + 1
        blur_ksize = max(1, blur_ksize)
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray = img.copy()  # 假设输入已经是灰度图或单通道图

    # 从参数获取 HoughCircles 设置
    dp = params.get('dp', 1.2)
    minDist = params.get('minDist', 20)
    param1 = params.get('param1', 50)  # Canny 边缘检测的高阈值
    param2 = params.get('param2', 30)  # 圆心检测的累加器阈值
    minRadius = params.get('minRadius', 0)
    maxRadius = params.get('maxRadius', 0)

    try:
        # 执行霍夫圆检测
        # 注意：参数需要是整数，dp除外
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, float(dp), int(minDist),
                                   param1=int(param1), param2=int(param2),
                                   minRadius=int(minRadius), maxRadius=int(maxRadius))

        # 准备输出图像 (确保是彩色以便绘制)
        if len(img.shape) == 3:
            output_img = img.copy()
        else:
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 如果检测到圆
        if circles is not None:
            circles = np.uint16(np.around(circles))  # 转换为整数坐标和半径
            color_b = params.get('Color_B', 0)
            color_g = params.get('Color_G', 255)
            color_r = params.get('Color_R', 0)
            thickness = params.get('Thickness', 2)

            # 绘制检测到的圆
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                # 绘制圆心
                # cv2.circle(output_img, center, 1, (0, 100, 100), 2)
                # 绘制圆轮廓
                cv2.circle(output_img, center, radius, (int(color_b), int(color_g), int(color_r)), int(thickness))

        return output_img

    except cv2.error as e:
        print(f"OpenCV Error in Hough Circles: {e}")
        h, w = img.shape[:2]
        error_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # UI: 界面显示中文错误提示
        return cv2.putText(error_img, "霍夫圆检测错误", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    except Exception as e:
        print(f"Error in Hough Circles processing: {e}")
        h, w = img.shape[:2]
        error_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # UI: 界面显示中文错误提示
        return cv2.putText(error_img, f"处理错误: {e}", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process_image_contours(img, params):
    """ Finds and draws contours """
    prep_method = params.get('Preprocessing', 'Canny')
    canny_thresh1 = params.get('Canny_Thresh1', 50)
    canny_thresh2 = params.get('Canny_Thresh2', 150)
    thresh_val = params.get('Binary_Thresh', 127)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()  # Work on a copy

    try:
        if prep_method == 'Canny':
            processed_for_contours = cv2.Canny(gray, int(canny_thresh1), int(canny_thresh2))
        elif prep_method == 'Threshold':
            _, processed_for_contours = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)
        else:  # Default to Canny
            processed_for_contours = cv2.Canny(gray, int(canny_thresh1), int(canny_thresh2))

        # Find Contours
        mode_str = params.get('Mode', 'RETR_TREE')
        method_str = params.get('Method', 'CHAIN_APPROX_SIMPLE')
        mode_map = {'RETR_EXTERNAL': cv2.RETR_EXTERNAL, 'RETR_LIST': cv2.RETR_LIST, 'RETR_CCOMP': cv2.RETR_CCOMP,
                    'RETR_TREE': cv2.RETR_TREE}
        method_map = {'CHAIN_APPROX_NONE': cv2.CHAIN_APPROX_NONE, 'CHAIN_APPROX_SIMPLE': cv2.CHAIN_APPROX_SIMPLE}
        mode = mode_map.get(mode_str, cv2.RETR_TREE)
        method = method_map.get(method_str, cv2.CHAIN_APPROX_SIMPLE)

        contours, hierarchy = cv2.findContours(processed_for_contours, mode, method)

        # Draw Contours
        draw_index = params.get('Draw_Index', -1)  # -1 draws all
        color_b = params.get('Color_B', 0)
        color_g = params.get('Color_G', 255)
        color_r = params.get('Color_R', 0)
        thickness = params.get('Thickness', 1)

        # Draw on a copy of the original image to preserve colors
        # Ensure output_img is color if the original was, or create one if drawing on gray
        if len(img.shape) == 3:
            output_img = img.copy()
        else:
            # Convert single channel input (like mask) to BGR for colored contour drawing
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(output_img, contours, int(draw_index), (int(color_b), int(color_g), int(color_r)),
                         int(thickness))
        return output_img
    except cv2.error as e:
        # 注意：错误信息文本保持英文
        print(f"OpenCV Error in contour processing: {e}")
        # Return original image or an error image on failure
        h, w = img.shape[:2]
        error_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.putText(error_img, "Contour Error", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process_image_shi_tomasi(img, params):
    """ Detects corners using Shi-Tomasi algorithm """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    max_corners = params.get('Max_Corners', 100)
    quality_level = params.get('Quality_Level', 0.01)
    min_distance = params.get('Min_Distance', 10)
    block_size = params.get('Block_Size', 3)  # Added Block Size
    use_harris = params.get('Use_Harris', 0)  # Added Use Harris Detector option
    k = params.get('k_Harris', 0.04)  # Added Harris free parameter

    try:
        corners = cv2.goodFeaturesToTrack(gray, int(max_corners), float(quality_level), int(min_distance),
                                          blockSize=int(block_size), useHarrisDetector=bool(use_harris), k=float(k))

        # Ensure output is color for drawing colored circles
        if len(img.shape) == 3:
            output_img = img.copy()
        else:
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if corners is not None:
            corners = np.intp(corners)  # Use np.intp for indexing
            radius = params.get('Corner_Radius', 3)
            color_b = params.get('Color_B', 0)
            color_g = params.get('Color_G', 0)
            color_r = params.get('Color_R', 255)
            thickness = params.get('Thickness', -1)  # -1 fills the circle

            for i in corners:
                x, y = i.ravel()
                cv2.circle(output_img, (x, y), int(radius), (int(color_b), int(color_g), int(color_r)), int(thickness))
        return output_img
    except cv2.error as e:
        # 注意：错误信息文本保持英文
        print(f"OpenCV Error in Shi-Tomasi: {e}")
        h, w = img.shape[:2]
        error_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.putText(error_img, "Corner Error", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process_image_hough_lines(img, params):
    """ Detects lines using Probabilistic Hough Transform """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    canny_thresh1 = params.get('Canny_Thresh1', 50)
    canny_thresh2 = params.get('Canny_Thresh2', 150)
    aperture_size = params.get('Aperture_Size', 3)  # Added Aperture Size for Canny

    # Ensure aperture size is odd and >= 3
    aperture_size = int(aperture_size)
    if aperture_size < 3: aperture_size = 3
    if aperture_size % 2 == 0: aperture_size += 1

    try:
        edges = cv2.Canny(gray, int(canny_thresh1), int(canny_thresh2), apertureSize=aperture_size)

        rho = params.get('Rho', 1)
        theta_deg = params.get('Theta_Degrees', 1)  # Input theta in degrees for simplicity
        threshold = params.get('Threshold', 50)
        min_line_length = params.get('Min_Line_Length', 50)
        max_line_gap = params.get('Max_Line_Gap', 10)

        theta_rad = np.deg2rad(float(theta_deg))  # Convert degrees to radians for OpenCV

        lines = cv2.HoughLinesP(edges, float(rho), theta_rad, int(threshold),
                                minLineLength=int(min_line_length), maxLineGap=int(max_line_gap))

        # Ensure output is color for drawing colored lines
        if len(img.shape) == 3:
            output_img = img.copy()
        else:
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            color_b = params.get('Color_B', 0)
            color_g = params.get('Color_G', 0)
            color_r = params.get('Color_R', 255)
            thickness = params.get('Thickness', 1)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output_img, (x1, y1), (x2, y2), (int(color_b), int(color_g), int(color_r)), int(thickness))
        return output_img

    except cv2.error as e:
        # 注意：错误信息文本保持英文
        print(f"OpenCV Error in Hough Lines: {e}")
        h, w = img.shape[:2]
        error_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.putText(error_img, "Hough Error", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process_image_canny(img, params):
    """ Applies Canny edge detection """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    threshold1 = params.get('threshold1', 50)
    threshold2 = params.get('threshold2', 150)
    aperture_size = params.get('apertureSize', 3)
    L2gradient = params.get('L2gradient', 0)  # Boolean (0 or 1)

    # Ensure aperture size is odd and between 3 and 7
    aperture_size = int(aperture_size)
    if aperture_size < 3:
        aperture_size = 3
    elif aperture_size > 7:
        aperture_size = 7
    if aperture_size % 2 == 0: aperture_size += 1  # Make odd

    try:
        return cv2.Canny(gray, int(threshold1), int(threshold2),
                         apertureSize=aperture_size, L2gradient=bool(L2gradient))
    except cv2.error as e:
        # 注意：错误信息文本保持英文
        print(f"OpenCV Error in Canny: {e}")
        h, w = gray.shape[:2]
        return cv2.putText(gray, "Canny Error", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def process_image_threshold(img, params):
    """ Applies image thresholding """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    thresh = params.get('thresh', 127)
    maxval = params.get('maxval', 255)
    thresh_type_str = params.get('type', 'THRESH_BINARY')
    thresh_type_map = {
        'THRESH_BINARY': cv2.THRESH_BINARY, 'THRESH_BINARY_INV': cv2.THRESH_BINARY_INV,
        'THRESH_TRUNC': cv2.THRESH_TRUNC, 'THRESH_TOZERO': cv2.THRESH_TOZERO,
        'THRESH_TOZERO_INV': cv2.THRESH_TOZERO_INV,
        'THRESH_OTSU': cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        'THRESH_TRIANGLE': cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    }
    thresh_type = thresh_type_map.get(thresh_type_str, cv2.THRESH_BINARY)

    try:
        if 'OTSU' in thresh_type_str or 'TRIANGLE' in thresh_type_str:
            ret, processed_img = cv2.threshold(gray, 0, int(maxval), thresh_type)
        else:
            ret, processed_img = cv2.threshold(gray, int(thresh), int(maxval), thresh_type)
        return processed_img
    except cv2.error as e:
        # 注意：错误信息文本保持英文
        print(f"OpenCV Error in Threshold: {e}")
        h, w = gray.shape[:2]
        return cv2.putText(gray, "Threshold Error", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def process_image_gaussian_blur(img, params):
    """ Applies Gaussian blur """
    ksize_w = params.get('ksize_w', 5)
    ksize_h = params.get('ksize_h', 5)
    sigmaX = params.get('sigmaX', 0)
    sigmaY = params.get('sigmaY', 0)  # Added sigmaY
    borderType = params.get('borderType', 'BORDER_DEFAULT')  # Added borderType

    # Kernel size must be positive and odd
    ksize_w = int(ksize_w) if int(ksize_w) % 2 != 0 else int(ksize_w) + 1
    ksize_h = int(ksize_h) if int(ksize_h) % 2 != 0 else int(ksize_h) + 1
    ksize_w = max(1, ksize_w)
    ksize_h = max(1, ksize_h)

    border_map = {'BORDER_DEFAULT': cv2.BORDER_DEFAULT, 'BORDER_CONSTANT': cv2.BORDER_CONSTANT,
                  'BORDER_REPLICATE': cv2.BORDER_REPLICATE, 'BORDER_REFLECT': cv2.BORDER_REFLECT,
                  'BORDER_WRAP': cv2.BORDER_WRAP, 'BORDER_REFLECT_101': cv2.BORDER_REFLECT_101}
    border = border_map.get(borderType, cv2.BORDER_DEFAULT)

    try:
        return cv2.GaussianBlur(img, (ksize_w, ksize_h), float(sigmaX), sigmaY=float(sigmaY), borderType=border)
    except cv2.error as e:
        # 注意：错误信息文本保持英文
        print(f"OpenCV Error in Gaussian Blur: {e}")
        h, w = img.shape[:2]
        error_img = img.copy()
        return cv2.putText(error_img, "Blur Error", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process_image_dilate(img, params):
    """ Applies dilation operation """
    kernel_size = params.get('kernel_size', 3)
    iterations = params.get('iterations', 1)
    kernel_shape_str = params.get('kernel_shape', 'MORPH_RECT')

    shape_map = {'MORPH_RECT': cv2.MORPH_RECT, 'MORPH_CROSS': cv2.MORPH_CROSS, 'MORPH_ELLIPSE': cv2.MORPH_ELLIPSE}
    kernel_shape = shape_map.get(kernel_shape_str, cv2.MORPH_RECT)
    ksize = int(kernel_size)
    kernel = cv2.getStructuringElement(kernel_shape, (ksize, ksize))

    try:
        return cv2.dilate(img, kernel, iterations=int(iterations))
    except cv2.error as e:
        # 注意：错误信息文本保持英文
        print(f"OpenCV Error in Dilate: {e}")
        h, w = img.shape[:2]
        error_img = img.copy()
        return cv2.putText(error_img, "Dilate Error", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process_image_erode(img, params):
    """ Applies erosion operation """
    kernel_size = params.get('kernel_size', 3)
    iterations = params.get('iterations', 1)
    kernel_shape_str = params.get('kernel_shape', 'MORPH_RECT')

    shape_map = {'MORPH_RECT': cv2.MORPH_RECT, 'MORPH_CROSS': cv2.MORPH_CROSS, 'MORPH_ELLIPSE': cv2.MORPH_ELLIPSE}
    kernel_shape = shape_map.get(kernel_shape_str, cv2.MORPH_RECT)
    ksize = int(kernel_size)
    kernel = cv2.getStructuringElement(kernel_shape, (ksize, ksize))

    try:
        return cv2.erode(img, kernel, iterations=int(iterations))
    except cv2.error as e:
        # 注意：错误信息文本保持英文
        print(f"OpenCV Error in Erode: {e}")
        h, w = img.shape[:2]
        error_img = img.copy()
        return cv2.putText(error_img, "Erode Error", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# --- END: Functions from the prompt ---


# --- 函数和参数定义 (使用更多选项更新) ---
# 注意：这里的字典键和参数名保持英文，以确保代码逻辑正确
OPENCV_FUNCTIONS = {
    "Canny Edge Detection": {
        "function": process_image_canny,
        "params": {
            "threshold1": {"type": "slider", "range": (0, 255, 50)},
            "threshold2": {"type": "slider", "range": (0, 255, 150)},
            "apertureSize": {"type": "dropdown", "options": [3, 5, 7], "default": 3},  # Must be odd: 3, 5, or 7
            "L2gradient": {"type": "checkbox", "default": 0},  # 0 for False, 1 for True
        },
        "code_template": "# Ensure image is grayscale if needed\nif len(img.shape) == 3:\n    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nelse:\n    gray = img\nprocessed_img = cv2.Canny(gray, threshold1={threshold1}, threshold2={threshold2}, apertureSize={apertureSize}, L2gradient=bool({L2gradient}))"
    },
    "Thresholding": {
        "function": process_image_threshold,
        "params": {
            "thresh": {"type": "slider", "range": (0, 255, 127)},
            "maxval": {"type": "slider", "range": (0, 255, 255)},
            "type": {"type": "dropdown",
                     "options": ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO',
                                 'THRESH_TOZERO_INV', 'THRESH_OTSU', 'THRESH_TRIANGLE'], "default": 'THRESH_BINARY'}
        },
        "code_template": "# Ensure image is grayscale if needed\nif len(img.shape) == 3:\n    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nelse:\n    gray = img\nthresh_type_map = {{'THRESH_BINARY': cv2.THRESH_BINARY, 'THRESH_BINARY_INV': cv2.THRESH_BINARY_INV, 'THRESH_TRUNC': cv2.THRESH_TRUNC, 'THRESH_TOZERO': cv2.THRESH_TOZERO, 'THRESH_TOZERO_INV': cv2.THRESH_TOZERO_INV, 'THRESH_OTSU': cv2.THRESH_BINARY + cv2.THRESH_OTSU, 'THRESH_TRIANGLE': cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE}}\nthresh_type = thresh_type_map.get('{type}', cv2.THRESH_BINARY)\n# Otsu/Triangle ignore the threshold value, others use it\nif 'OTSU' in '{type}' or 'TRIANGLE' in '{type}':\n    ret, processed_img = cv2.threshold(gray, 0, {maxval}, thresh_type)\nelse:\n    ret, processed_img = cv2.threshold(gray, {thresh}, {maxval}, thresh_type)"
    },
    "HSV Color Threshold": {
        "function": process_image_hsv_threshold,
        "params": {
            "H_Lower": {"type": "slider", "range": (0, 179, 0)},  # Hue range is 0-179 in OpenCV
            "H_Upper": {"type": "slider", "range": (0, 179, 179)},
            "S_Lower": {"type": "slider", "range": (0, 255, 50)},
            "S_Upper": {"type": "slider", "range": (0, 255, 255)},
            "V_Lower": {"type": "slider", "range": (0, 255, 50)},
            "V_Upper": {"type": "slider", "range": (0, 255, 255)},
        },
        "code_template": "# Input 'img' must be BGR color\nif len(img.shape) < 3 or img.shape[2] != 3:\n    print('HSV Thresholding requires a color image.')\n    # Handle error appropriately, e.g., return original or blank image\n    processed_img = img\nelse:\n    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\n    lower_bound = np.array([{H_Lower}, {S_Lower}, {V_Lower}])\n    upper_bound = np.array([{H_Upper}, {S_Upper}, {V_Upper}])\n    processed_img = cv2.inRange(hsv, lower_bound, upper_bound)\n    # Optional: Mask original image\n    # result = cv2.bitwise_and(img, img, mask=processed_img)"
    },
    "Gaussian Blur": {
        "function": process_image_gaussian_blur,
        "params": {
            "ksize_w": {"type": "slider", "range": (1, 51, 5), "step": 2},  # Step 2 to suggest odd numbers
            "ksize_h": {"type": "slider", "range": (1, 51, 5), "step": 2},
            "sigmaX": {"type": "slider", "range": (0, 50, 0)},
            "sigmaY": {"type": "slider", "range": (0, 50, 0)},  # Optional: if 0, same as sigmaX
            "borderType": {"type": "dropdown",
                           "options": ['BORDER_DEFAULT', 'BORDER_CONSTANT', 'BORDER_REPLICATE', 'BORDER_REFLECT',
                                       'BORDER_WRAP', 'BORDER_REFLECT_101'], "default": 'BORDER_DEFAULT'}
        },
        "code_template": "# Ensure kernel sizes are odd\nksize_w = {ksize_w} if {ksize_w} % 2 != 0 else {ksize_w} + 1\nksize_h = {ksize_h} if {ksize_h} % 2 != 0 else {ksize_h} + 1\nborder_map = {{'BORDER_DEFAULT': cv2.BORDER_DEFAULT, 'BORDER_CONSTANT': cv2.BORDER_CONSTANT, 'BORDER_REPLICATE': cv2.BORDER_REPLICATE, 'BORDER_REFLECT': cv2.BORDER_REFLECT, 'BORDER_WRAP': cv2.BORDER_WRAP, 'BORDER_REFLECT_101': cv2.BORDER_REFLECT_101}}\nborder = border_map.get('{borderType}', cv2.BORDER_DEFAULT)\nprocessed_img = cv2.GaussianBlur(img, (ksize_w, ksize_h), {sigmaX}, sigmaY={sigmaY}, borderType=border)"
    },
    "Dilation": {
        "function": process_image_dilate,
        "params": {
            "kernel_shape": {"type": "dropdown", "options": ['MORPH_RECT', 'MORPH_CROSS', 'MORPH_ELLIPSE'],
                             "default": 'MORPH_RECT'},
            "kernel_size": {"type": "slider", "range": (1, 21, 3)},
            "iterations": {"type": "slider", "range": (1, 10, 1)},
        },
        "code_template": "shape_map = {{'MORPH_RECT': cv2.MORPH_RECT, 'MORPH_CROSS': cv2.MORPH_CROSS, 'MORPH_ELLIPSE': cv2.MORPH_ELLIPSE}}\nkernel_shape = shape_map.get('{kernel_shape}', cv2.MORPH_RECT)\nkernel = cv2.getStructuringElement(kernel_shape, ({kernel_size}, {kernel_size}))\nprocessed_img = cv2.dilate(img, kernel, iterations={iterations})"
    },
    "Erosion": {
        "function": process_image_erode,
        "params": {
            "kernel_shape": {"type": "dropdown", "options": ['MORPH_RECT', 'MORPH_CROSS', 'MORPH_ELLIPSE'],
                             "default": 'MORPH_RECT'},
            "kernel_size": {"type": "slider", "range": (1, 21, 3)},
            "iterations": {"type": "slider", "range": (1, 10, 1)},
        },
        "code_template": "shape_map = {{'MORPH_RECT': cv2.MORPH_RECT, 'MORPH_CROSS': cv2.MORPH_CROSS, 'MORPH_ELLIPSE': cv2.MORPH_ELLIPSE}}\nkernel_shape = shape_map.get('{kernel_shape}', cv2.MORPH_RECT)\nkernel = cv2.getStructuringElement(kernel_shape, ({kernel_size}, {kernel_size}))\nprocessed_img = cv2.erode(img, kernel, iterations={iterations})"
    },
    "Contour Detection": {
        "function": process_image_contours,
        "params": {
            "Preprocessing": {"type": "dropdown", "options": ['Canny', 'Threshold'], "default": 'Canny'},
            "Canny_Thresh1": {"type": "slider", "range": (0, 255, 50)},
            "Canny_Thresh2": {"type": "slider", "range": (0, 255, 150)},
            "Binary_Thresh": {"type": "slider", "range": (0, 255, 127)},
            "Mode": {"type": "dropdown", "options": ['RETR_EXTERNAL', 'RETR_LIST', 'RETR_CCOMP', 'RETR_TREE'],
                     "default": 'RETR_TREE'},
            "Method": {"type": "dropdown", "options": ['CHAIN_APPROX_NONE', 'CHAIN_APPROX_SIMPLE'],
                       "default": 'CHAIN_APPROX_SIMPLE'},
            "Draw_Index": {"type": "slider", "range": (-1, 100, -1)},  # -1 for all
            "Color_B": {"type": "slider", "range": (0, 255, 0)},
            "Color_G": {"type": "slider", "range": (0, 255, 255)},
            "Color_R": {"type": "slider", "range": (0, 255, 0)},
            "Thickness": {"type": "slider", "range": (-1, 10, 1)},  # -1 for fill
        },
        "code_template": "# --- Contour Detection Code ---\n# 1. Preprocessing\nif len(img.shape) == 3:\n    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nelse:\n    gray = img.copy()\nif '{Preprocessing}' == 'Canny':\n    processed_for_contours = cv2.Canny(gray, {Canny_Thresh1}, {Canny_Thresh2})\nelif '{Preprocessing}' == 'Threshold':\n    _, processed_for_contours = cv2.threshold(gray, {Binary_Thresh}, 255, cv2.THRESH_BINARY)\nelse:\n    processed_for_contours = cv2.Canny(gray, {Canny_Thresh1}, {Canny_Thresh2})\n# 2. Find Contours\nmode_map = {{'RETR_EXTERNAL': cv2.RETR_EXTERNAL, 'RETR_LIST': cv2.RETR_LIST, 'RETR_CCOMP': cv2.RETR_CCOMP, 'RETR_TREE': cv2.RETR_TREE}}\nmethod_map = {{'CHAIN_APPROX_NONE': cv2.CHAIN_APPROX_NONE, 'CHAIN_APPROX_SIMPLE': cv2.CHAIN_APPROX_SIMPLE}}\nmode = mode_map.get('{Mode}', cv2.RETR_TREE)\nmethod = method_map.get('{Method}', cv2.CHAIN_APPROX_SIMPLE)\ncontours, hierarchy = cv2.findContours(processed_for_contours, mode, method)\n# 3. Draw Contours\nif len(img.shape) == 3:\n    output_img = img.copy()\nelse:\n    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Draw on color image\ncolor = ({Color_B}, {Color_G}, {Color_R})\ncv2.drawContours(output_img, contours, {Draw_Index}, color, {Thickness})\nprocessed_img = output_img"
    },
    "Shi-Tomasi Corners": {
        "function": process_image_shi_tomasi,
        "params": {
            "Max_Corners": {"type": "slider", "range": (1, 500, 100)},
            "Quality_Level": {"type": "float_slider", "range": (0.01, 1.0, 0.01)},  # Float slider
            "Min_Distance": {"type": "slider", "range": (1, 100, 10)},
            "Block_Size": {"type": "slider", "range": (3, 31, 3), "step": 2},  # Odd numbers usually
            "Use_Harris": {"type": "checkbox", "default": 0},
            "k_Harris": {"type": "float_slider", "range": (0.01, 0.2, 0.04)},  # Harris free parameter
            "Corner_Radius": {"type": "slider", "range": (1, 10, 3)},
            "Color_B": {"type": "slider", "range": (0, 255, 0)},
            "Color_G": {"type": "slider", "range": (0, 255, 0)},
            "Color_R": {"type": "slider", "range": (0, 255, 255)},
            "Thickness": {"type": "slider", "range": (-1, 5, -1)},  # -1 fills circle
        },
        "code_template": "# --- Shi-Tomasi Corner Detection ---\nif len(img.shape) == 3:\n    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nelse:\n    gray = img.copy()\ncorners = cv2.goodFeaturesToTrack(gray, {Max_Corners}, {Quality_Level:.4f}, {Min_Distance}, blockSize={Block_Size}, useHarrisDetector=bool({Use_Harris}), k={k_Harris:.4f})\nif len(img.shape) == 3:\n    output_img = img.copy()\nelse:\n    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)\nif corners is not None:\n    corners = np.intp(corners)\n    color = ({Color_B}, {Color_G}, {Color_R})\n    for i in corners:\n        x, y = i.ravel()\n        cv2.circle(output_img, (x, y), {Corner_Radius}, color, {Thickness})\nprocessed_img = output_img"
    },
    "Hough Lines (Probabilistic)": {
        "function": process_image_hough_lines,
        "params": {
            "Canny_Thresh1": {"type": "slider", "range": (0, 255, 50)},
            "Canny_Thresh2": {"type": "slider", "range": (0, 255, 150)},
            "Aperture_Size": {"type": "dropdown", "options": [3, 5, 7], "default": 3},  # For Canny pre-step
            "Rho": {"type": "slider", "range": (1, 10, 1)},  # Distance resolution
            "Theta_Degrees": {"type": "slider", "range": (1, 90, 1)},  # Angle resolution in degrees
            "Threshold": {"type": "slider", "range": (10, 300, 50)},  # Accumulator threshold
            "Min_Line_Length": {"type": "slider", "range": (5, 200, 50)},
            "Max_Line_Gap": {"type": "slider", "range": (1, 100, 10)},
            "Color_B": {"type": "slider", "range": (0, 255, 0)},
            "Color_G": {"type": "slider", "range": (0, 255, 0)},
            "Color_R": {"type": "slider", "range": (0, 255, 255)},
            "Thickness": {"type": "slider", "range": (1, 10, 1)},
        },
        "code_template": "# --- Hough Line Detection (Probabilistic) ---\nif len(img.shape) == 3:\n    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nelse:\n    gray = img.copy()\nedges = cv2.Canny(gray, {Canny_Thresh1}, {Canny_Thresh2}, apertureSize={Aperture_Size})\ntheta_rad = np.deg2rad({Theta_Degrees})\nlines = cv2.HoughLinesP(edges, {Rho}, theta_rad, {Threshold}, minLineLength={Min_Line_Length}, maxLineGap={Max_Line_Gap})\nif len(img.shape) == 3:\n    output_img = img.copy()\nelse:\n    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)\nif lines is not None:\n    color = ({Color_B}, {Color_G}, {Color_R})\n    for line in lines:\n        x1, y1, x2, y2 = line[0]\n        cv2.line(output_img, (x1, y1), (x2, y2), color, {Thickness})\nprocessed_img = output_img"
    },
    # Add more functions here if needed
    "Hough Circle Detection": {
        "function": process_image_hough_circles,
        "params": {
            # 预处理参数
            "Blur_Kernel_Size": {"type": "slider", "range": (1, 25, 5), "step": 2},  # 模糊核大小 (奇数)
            # HoughCircles 参数
            "dp": {"type": "float_slider", "range": (1.0, 3.0, 1.2)},
            # 累加器分辨率 ( inversely proportional to image resolution)
            "minDist": {"type": "slider", "range": (1, 200, 20)},  # 检测到的圆的中心之间的最小距离
            "param1": {"type": "slider", "range": (10, 255, 50)},  # Canny边缘检测的高阈值
            "param2": {"type": "slider", "range": (10, 200, 30)},  # 累加器阈值，越小检测到的圆越多
            "minRadius": {"type": "slider", "range": (0, 200, 0)},  # 最小圆半径
            "maxRadius": {"type": "slider", "range": (0, 500, 0)},  # 最大圆半径 (0表示不限制)
            # 绘图参数
            "Color_B": {"type": "slider", "range": (0, 255, 0)},
            "Color_G": {"type": "slider", "range": (0, 255, 255)},
            "Color_R": {"type": "slider", "range": (0, 255, 0)},
            "Thickness": {"type": "slider", "range": (1, 10, 2)},  # 线条粗细 (不支持填充 -1)
        },
        "code_template": "# --- Hough Circle Detection Code ---\n# 1. Preprocessing (Grayscale and Blur)\nif len(img.shape) == 3:\n    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n    blur_ksize = {Blur_Kernel_Size} if {Blur_Kernel_Size} % 2 != 0 else {Blur_Kernel_Size} + 1\n    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)\nelse:\n    gray = img.copy()\n# 2. Hough Circle Detection\ncircles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp={dp:.2f}, minDist={minDist}, \n                           param1={param1}, param2={param2}, \n                           minRadius={minRadius}, maxRadius={maxRadius})\n# 3. Drawing\nif len(img.shape) == 3:\n    output_img = img.copy()\nelse:\n    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Draw on color image\nif circles is not None:\n    circles = np.uint16(np.around(circles))\n    color = ({Color_B}, {Color_G}, {Color_R})\n    for i in circles[0, :]:\n        center = (i[0], i[1])\n        radius = i[2]\n        cv2.circle(output_img, center, radius, color, {Thickness})\nprocessed_img = output_img"
    },
    # ... (如果后面还有其他函数，确保这里有逗号) ...
}


# --- Tkinter 应用程序类 ---

class ImageProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # --- UI: 窗口标题改为中文 ---
        self.title("OpenCV 可视化参数调节器")
        self.geometry("1300x850")  # 保持默认大小

        # 数据存储 (变量名保持英文)
        self.original_img_cv = None
        self.processed_img_cv = None
        self.display_orig_tk = None
        self.display_proc_tk = None
        self.current_params = {}
        self.param_widgets = {}
        self.param_vars = {}
        self.selected_func_name = None

        # --- 摄像头相关变量 (变量名保持英文) ---
        self.camera_active = False
        self.cap = None
        self.camera_index_var = tk.StringVar(value="0")
        self.update_job = None

        # --- UI 布局 ---
        self._setup_ui()

        # --- 协议绑定 ---
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        # --- 主框架 ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 顶部: 文件/摄像头操作和功能选择 ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # --- UI: 文件上传按钮文本改为中文 ---
        self.upload_button = ttk.Button(top_frame, text="上传图像", command=self._load_image)
        self.upload_button.pack(side=tk.LEFT, padx=(0, 5))

        # 摄像头控制框架
        cam_frame = ttk.Frame(top_frame)
        cam_frame.pack(side=tk.LEFT, padx=(10, 10))
        # --- UI: 摄像头按钮初始文本改为中文 ---
        # 注意：切换状态时的 "停止摄像头" 文本需要在 _toggle_camera 方法中修改（如果该方法存在）
        self.cam_button = ttk.Button(cam_frame, text="启动摄像头", command=self._toggle_camera, width=12)
        self.cam_button.pack(side=tk.LEFT)
        # --- UI: 摄像头索引标签文本改为中文 ---
        ttk.Label(cam_frame, text="索引:").pack(side=tk.LEFT, padx=(5, 2))
        self.cam_index_entry = ttk.Entry(cam_frame, textvariable=self.camera_index_var, width=3)
        self.cam_index_entry.pack(side=tk.LEFT)

        # 功能选择
        # --- UI: 功能标签文本改为中文 ---
        ttk.Label(top_frame, text="功能:").pack(side=tk.LEFT, padx=(10, 5))
        # 注意：Combobox 的值列表来自 OPENCV_FUNCTIONS 的键，保持英文
        self.func_combobox = ttk.Combobox(top_frame, values=list(OPENCV_FUNCTIONS.keys()), state="disabled", width=25)
        self.func_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.func_combobox.bind("<<ComboboxSelected>>", self._on_function_select)

        # --- UI: 保存处理后图像按钮文本改为中文 ---
        self.save_button = ttk.Button(top_frame, text="保存处理结果", command=self._save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))

        # --- UI: 保存/加载参数按钮文本改为中文 ---
        self.save_params_button = ttk.Button(top_frame, text="保存参数", command=self._save_params, state=tk.DISABLED)
        self.save_params_button.pack(side=tk.LEFT, padx=(0, 5))
        self.load_params_button = ttk.Button(top_frame, text="加载参数", command=self._load_params, state=tk.DISABLED)
        self.load_params_button.pack(side=tk.LEFT, padx=(0, 10))

        # --- 中部: 图像显示和参数调节 ---
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 参数控制区 (右侧)
        # --- UI: 参数区域标签文本改为中文 ---
        self.param_frame_container = ttk.LabelFrame(middle_frame, text="参数", padding="10")
        self.param_frame_container.pack(fill=tk.Y, side=tk.RIGHT, padx=(10, 0), anchor=tk.N)
        self.param_frame = ttk.Frame(self.param_frame_container)
        self.param_frame.pack(fill=tk.BOTH, expand=True)

        # 图像显示区 (左侧, PanedWindow)
        img_pane = ttk.PanedWindow(middle_frame, orient=tk.HORIZONTAL)
        img_pane.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # 左侧: 原始图像
        left_frame = ttk.Frame(img_pane, relief=tk.SUNKEN, borderwidth=1)
        # --- UI: 原始图像标签文本改为中文 ---
        ttk.Label(left_frame, text="原始图像").pack(pady=5)
        # --- UI: 原始图像区域初始提示文本改为中文 ---
        self.img_label_left = ttk.Label(left_frame, text="请上传图像或启动摄像头", anchor=tk.CENTER)
        self.img_label_left.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        img_pane.add(left_frame, weight=1)

        # 右侧: 处理后图像
        right_frame = ttk.Frame(img_pane, relief=tk.SUNKEN, borderwidth=1)
        # --- UI: 处理后图像标签文本改为中文 ---
        ttk.Label(right_frame, text="处理后图像").pack(pady=5)
        # --- UI: 处理后图像区域初始占位符保持不变 ---
        self.img_label_right = ttk.Label(right_frame, text="---", anchor=tk.CENTER)
        self.img_label_right.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        img_pane.add(right_frame, weight=1)

        # --- 底部: 代码生成 ---
        # --- UI: 代码区域标签文本改为中文 ---
        bottom_frame = ttk.LabelFrame(main_frame, text="生成的代码", padding=(10, 5))
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        # --- UI: 复制代码按钮文本改为中文 ---
        self.code_button = ttk.Button(bottom_frame, text="复制代码到剪贴板", command=self._copy_code_to_clipboard,
                                      state=tk.DISABLED)
        self.code_button.pack(side=tk.LEFT, padx=(0, 10), anchor='nw')

        # 使 Text 小部件可扩展
        code_text_frame = ttk.Frame(bottom_frame)
        code_text_frame.pack(fill=tk.BOTH, expand=True)

        self.code_text = tk.Text(code_text_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, relief=tk.SUNKEN,
                                 borderwidth=1, font=("Courier New", 9))
        scrollbar = ttk.Scrollbar(code_text_frame, orient=tk.VERTICAL, command=self.code_text.yview)
        self.code_text['yscrollcommand'] = scrollbar.set
        # Add scrollbar packing
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Pack text widget after scrollbar

    # --- 其他方法 (保持不变) ---

    def _load_image(self):
        # (此处省略具体实现，保持原样)
        # Example placeholder logic:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            self.original_img_cv = cv2.imread(file_path)
            if self.original_img_cv is None:
                raise ValueError("无法加载图像文件")
            print(f"Loaded image: {file_path}")  # 保持英文日志
            # Stop camera if active
            if self.camera_active:
                self._toggle_camera()  # This should handle UI update of cam button
            self._update_image_display(self.original_img_cv, self.img_label_left, 'display_orig_tk')
            self.func_combobox.config(state="readonly")  # Enable function selection
            self._clear_processed_image()  # Clear right panel
            # Trigger processing if a function is already selected
            if self.selected_func_name:
                self._apply_processing()
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {e}")  # 消息框文本可考虑翻译

    def _toggle_camera(self):
        # (此处省略具体实现，保持原样)
        # Example placeholder logic:
        if self.camera_active:
            self.camera_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.update_job:
                self.after_cancel(self.update_job)
                self.update_job = None
            # --- UI: 更新按钮文本为中文 ---
            self.cam_button.config(text="启动摄像头")
            self.upload_button.config(state=tk.NORMAL)
            self.cam_index_entry.config(state=tk.NORMAL)
            # Maybe load a default placeholder? Or clear left image?
            # self.img_label_left.config(image='') # Clear image
            # self.img_label_left.image = None
            # self.img_label_left.config(text="请上传图像或启动摄像头") # Reset text
            # self.original_img_cv = None
            # self._clear_processed_image()
            # self.func_combobox.config(state="disabled")
            print("Camera stopped.")  # 保持英文日志
        else:
            try:
                cam_index_str = self.camera_index_var.get()
                cam_index = int(cam_index_str)
                self.cap = cv2.VideoCapture(cam_index)
                if not self.cap.isOpened():
                    raise ValueError(f"无法打开摄像头索引 {cam_index}")
                self.camera_active = True
                # --- UI: 更新按钮文本为中文 ---
                self.cam_button.config(text="停止摄像头")
                self.upload_button.config(state=tk.DISABLED)
                self.cam_index_entry.config(state=tk.DISABLED)
                self.func_combobox.config(state="readonly")  # Enable function selection
                print(f"Camera {cam_index} started.")  # 保持英文日志
                self._update_camera_frame()  # Start the update loop
            except ValueError as e:
                messagebox.showerror("错误", f"启动摄像头失败: {e}")  # 消息框文本可考虑翻译
                if self.cap:
                    self.cap.release()
                self.cap = None
                self.camera_active = False

    def _update_camera_frame(self):
        if not self.camera_active or not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            self.original_img_cv = frame
            # Process frame immediately only if a function is selected
            self._update_image_display(self.original_img_cv, self.img_label_left, 'display_orig_tk')
            if self.selected_func_name:
                self._apply_processing(update_params=False)  # Avoid constant param reset

        # Schedule next update
        self.update_job = self.after(CAMERA_UPDATE_DELAY_MS, self._update_camera_frame)

    def _update_image_display(self, cv_img, label_widget, tk_photo_attr_name):
        if cv_img is None:
            label_widget.config(image='', text="---")  # Use placeholder
            setattr(self, tk_photo_attr_name, None)
            return

        try:
            # Resize for display
            h, w = cv_img.shape[:2]
            scale = min(MAX_IMAGE_WIDTH / w, MAX_IMAGE_HEIGHT / h)
            if scale < 1.0:  # Only downscale
                nw, nh = int(w * scale), int(h * scale)
                display_img = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                display_img = cv_img

            # Convert color space for Tkinter
            if len(display_img.shape) == 3:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(display_img)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update label
            label_widget.config(image=img_tk, text="")  # Clear text when showing image
            setattr(self, tk_photo_attr_name, img_tk)  # Keep reference

        except Exception as e:
            print(f"Error updating image display: {e}")  # 保持英文日志
            label_widget.config(image='', text="图像显示错误")  # UI: 错误提示改为中文
            setattr(self, tk_photo_attr_name, None)

    def _on_function_select(self, event=None):
        self.selected_func_name = self.func_combobox.get()
        if not self.selected_func_name:
            return

        # Clear previous parameter widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_widgets = {}
        self.param_vars = {}
        self.current_params = {}

        # Create new widgets based on the selected function
        func_info = OPENCV_FUNCTIONS.get(self.selected_func_name)
        if not func_info:
            print(f"Function '{self.selected_func_name}' not found in definitions.")  # 保持英文日志
            return

        params_def = func_info.get("params", {})
        row_index = 0
        for name, config in params_def.items():
            param_type = config.get("type")
            # Create Label - 参数名称保持英文，但可以在 UI 上显示翻译
            # 为了简单起见，这里仍然使用英文参数名作为标签
            label = ttk.Label(self.param_frame, text=f"{name}:")
            label.grid(row=row_index, column=0, sticky=tk.W, padx=5, pady=2)

            if param_type == "slider":
                min_val, max_val, default_val = config["range"]
                step = config.get("step", 1)  # Get step, default to 1
                var = tk.IntVar(value=int(default_val))
                self.param_vars[name] = var
                scale = ttk.Scale(self.param_frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL,
                                  command=lambda val, n=name: self._on_param_change(n, val, 'int'))
                entry = ttk.Entry(self.param_frame, textvariable=var, width=5,
                                  validate="key", validatecommand=(self.register(self._validate_int), '%P'))
                # Set scale resolution if step is defined (though ttk.Scale doesn't directly use it like tkinter.Scale resolution)
                # We handle step logic in the change handler if needed, but UI shows continuous slide for ttk
                scale.grid(row=row_index, column=1, sticky=tk.EW, padx=5)
                entry.grid(row=row_index, column=2, sticky=tk.W, padx=5)
                self.param_widgets[name] = {'scale': scale, 'entry': entry}
                # Bind entry changes back to slider and processing
                entry.bind("<Return>", lambda event, n=name: self._on_entry_change(n, 'int'))
                entry.bind("<FocusOut>", lambda event, n=name: self._on_entry_change(n, 'int'))
                self.current_params[name] = int(default_val)  # Initialize

            elif param_type == "float_slider":
                min_val, max_val, default_val = config["range"]
                # For float, use DoubleVar
                var = tk.DoubleVar(value=float(default_val))
                # Round default value display if needed
                var.set(round(default_val, 4))  # Display with precision
                self.param_vars[name] = var
                scale = ttk.Scale(self.param_frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL,
                                  command=lambda val, n=name: self._on_param_change(n, val, 'float'))
                entry = ttk.Entry(self.param_frame, textvariable=var, width=7,  # Slightly wider for floats
                                  validate="key", validatecommand=(self.register(self._validate_float), '%P'))
                scale.grid(row=row_index, column=1, sticky=tk.EW, padx=5)
                entry.grid(row=row_index, column=2, sticky=tk.W, padx=5)
                self.param_widgets[name] = {'scale': scale, 'entry': entry}
                entry.bind("<Return>", lambda event, n=name: self._on_entry_change(n, 'float'))
                entry.bind("<FocusOut>", lambda event, n=name: self._on_entry_change(n, 'float'))
                self.current_params[name] = float(default_val)  # Initialize


            elif param_type == "dropdown":
                options = config.get("options", [])
                default_val = config.get("default", options[0] if options else "")
                var = tk.StringVar(value=default_val)
                self.param_vars[name] = var
                combobox = ttk.Combobox(self.param_frame, textvariable=var, values=options, state="readonly", width=18)
                combobox.grid(row=row_index, column=1, columnspan=2, sticky=tk.EW, padx=5)
                combobox.bind("<<ComboboxSelected>>",
                              lambda event, n=name: self._on_param_change(n, self.param_vars[n].get(), 'str'))
                self.param_widgets[name] = combobox
                self.current_params[name] = default_val  # Initialize

            elif param_type == "checkbox":
                default_val = config.get("default", 0)  # 0 or 1
                var = tk.BooleanVar(value=bool(default_val))
                self.param_vars[name] = var
                checkbox = ttk.Checkbutton(self.param_frame, variable=var,
                                           command=lambda n=name: self._on_param_change(n, var.get(), 'bool'))
                checkbox.grid(row=row_index, column=1, columnspan=2, sticky=tk.W, padx=5)
                self.param_widgets[name] = checkbox
                self.current_params[name] = bool(default_val)  # Initialize

            row_index += 1

        self.param_frame.columnconfigure(1, weight=1)  # Make slider column expandable

        # Enable controls that depend on function selection
        self.save_params_button.config(state=tk.NORMAL)
        self.load_params_button.config(state=tk.NORMAL)  # Enable loading anytime
        self.code_button.config(state=tk.NORMAL)

        # Apply initial processing with default parameters
        if self.original_img_cv is not None:
            self._apply_processing()
            self._generate_code()  # Generate code on function select
            self.save_button.config(state=tk.NORMAL if self.processed_img_cv is not None else tk.DISABLED)

    def _on_param_change(self, name, value, type_hint):
        # Called primarily by sliders and dropdowns/checkboxes
        try:
            # Convert value based on hint
            if type_hint == 'int':
                new_value = int(float(value))  # Sliders pass float, convert to int
                # Update Entry if it exists and differs
                if name in self.param_widgets and 'entry' in self.param_widgets[name]:
                    self.param_vars[name].set(new_value)
            elif type_hint == 'float':
                new_value = float(value)
                # Update Entry with rounded value
                if name in self.param_widgets and 'entry' in self.param_widgets[name]:
                    # Round to reasonable precision for display
                    display_val = round(new_value, 4)
                    self.param_vars[name].set(display_val)
                # Keep potentially higher precision for processing
                # Or decide to always use the rounded value:
                # new_value = round(new_value, 4) # Option: use rounded value internally too
            elif type_hint == 'bool':
                new_value = bool(value)  # Checkbox passes boolean
            else:  # string (dropdown)
                new_value = str(value)

            # Update only if value actually changed to avoid redundant processing
            if name not in self.current_params or self.current_params[name] != new_value:
                self.current_params[name] = new_value
                # print(f"Param '{name}' changed to: {new_value}") # Debug
                self._apply_processing()
                self._generate_code()

        except ValueError:
            print(f"Invalid value '{value}' for parameter '{name}'.")  # 保持英文日志
            # Optionally revert the widget value here if needed

    def _on_entry_change(self, name, type_hint):
        # Called when Entry loses focus or Enter is pressed
        if name not in self.param_vars: return
        var = self.param_vars[name]
        try:
            value_str = var.get()
            # Convert value based on hint and update slider/processing
            if type_hint == 'int':
                new_value = int(value_str)
                # Clamp value to slider range if applicable
                if name in self.param_widgets and 'scale' in self.param_widgets[name]:
                    scale = self.param_widgets[name]['scale']
                    min_val, max_val = scale.cget("from"), scale.cget("to")
                    new_value = max(int(min_val), min(int(max_val), new_value))
                    var.set(new_value)  # Update entry text if clamped
            elif type_hint == 'float':
                new_value = float(value_str)
                # Clamp value to slider range if applicable
                if name in self.param_widgets and 'scale' in self.param_widgets[name]:
                    scale = self.param_widgets[name]['scale']
                    min_val, max_val = scale.cget("from"), scale.cget("to")
                    new_value = max(float(min_val), min(float(max_val), new_value))
                    var.set(round(new_value, 4))  # Update entry text if clamped, with rounding
                    new_value = round(new_value, 4)  # Use rounded value for consistency
            else:
                # Should not happen for entry-linked params currently
                print(f"Unexpected type hint '{type_hint}' for entry change.")  # 保持英文日志
                return

            # Trigger update via _on_param_change to handle processing & code gen
            self._on_param_change(name, new_value, type_hint)

        except ValueError:
            # Invalid input in entry, potentially revert to last valid value
            print(f"Invalid input '{value_str}' for {name}. Reverting.")  # 保持英文日志
            var.set(self.current_params[name])  # Revert display

    def _validate_int(self, P):
        # Validation command for integer entries
        if P == "" or P == "-": return True  # Allow empty or just minus
        try:
            int(P)
            return True
        except ValueError:
            return False

    def _validate_float(self, P):
        # Validation command for float entries
        if P == "" or P == "-" or P == "." or P == "-.": return True  # Allow empty or partial floats
        try:
            float(P)
            return True
        except ValueError:
            return False

    def _apply_processing(self, update_params=True):
        if self.original_img_cv is None or self.selected_func_name is None:
            # Clear processed image if no original or no function
            self._clear_processed_image()
            return

        func_info = OPENCV_FUNCTIONS.get(self.selected_func_name)
        if not func_info: return

        func = func_info["function"]

        # Optionally update self.current_params from widgets just before processing
        # This ensures the latest values are used, especially if triggered not by param change
        if update_params:
            for name, var in self.param_vars.items():
                # Determine type from param_widgets or config if needed
                # Simple approach based on tk variable type:
                if isinstance(var, tk.IntVar):
                    self.current_params[name] = var.get()
                elif isinstance(var, tk.DoubleVar):
                    self.current_params[name] = var.get()
                    # Maybe round here too if consistency is desired
                    # self.current_params[name] = round(var.get(), 4)
                elif isinstance(var, tk.StringVar):
                    self.current_params[name] = var.get()
                elif isinstance(var, tk.BooleanVar):
                    self.current_params[name] = var.get()

        try:
            # Start timer
            # start_time = time.time()

            # Make a copy of the current parameters to pass to the function
            params_to_use = self.current_params.copy()
            # print("Applying processing with params:", params_to_use) # Debug

            # Execute the selected OpenCV processing function
            self.processed_img_cv = func(self.original_img_cv, params_to_use)

            # End timer
            # end_time = time.time()
            # print(f"Processing time: {end_time - start_time:.4f} seconds") # Debug

            # Update the display for the processed image
            self._update_image_display(self.processed_img_cv, self.img_label_right, 'display_proc_tk')
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            # Display error on the processed image label?
            error_text = f"处理错误:\n{e}"  # UI: 错误提示改为中文
            print(f"Error during processing '{self.selected_func_name}': {e}")  # 保持英文日志
            # Create a blank image with error text
            h, w = self.original_img_cv.shape[:2]
            error_img = np.zeros((h, w, 3), dtype=np.uint8)  # Black BGR
            # Put text on the error image (handling potential multi-line)
            y0, dy = 30, 30
            for i, line in enumerate(error_text.split('\n')):
                y = y0 + i * dy
                cv2.putText(error_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red text

            self.processed_img_cv = error_img  # Store the error image
            self._update_image_display(self.processed_img_cv, self.img_label_right, 'display_proc_tk')
            self.save_button.config(state=tk.DISABLED)  # Can't save error overlay easily

    def _clear_processed_image(self):
        self.processed_img_cv = None
        self.img_label_right.config(image='', text="---")  # Reset to placeholder
        self.display_proc_tk = None
        self.save_button.config(state=tk.DISABLED)

    def _generate_code(self):
        if not self.selected_func_name:
            self._clear_code_text()
            return

        func_info = OPENCV_FUNCTIONS.get(self.selected_func_name)
        if not func_info or "code_template" not in func_info:
            self._clear_code_text("Code template not available.")  # 保持英文提示
            return

        template = func_info["code_template"]
        params_for_code = self.current_params.copy()

        # Format float values in params if needed (e.g., for specific precision)
        for name, value in params_for_code.items():
            if isinstance(value, float):
                # Find if this param was defined as float_slider to apply formatting
                param_config = func_info["params"].get(name, {})
                if param_config.get("type") == "float_slider":
                    # Example: format floats to 4 decimal places in the code string
                    # Adjust precision as needed. The f-string format specifier handles this.
                    pass  # Let the f-string formatting in the template handle it.
                # else: could handle other float types if necessary

        try:
            # Use f-string formatting if template is designed for it,
            # otherwise use .format() if template uses {} placeholders.
            # The provided templates use {} style placeholders.
            generated_code = template.format(**params_for_code)

            # Add common imports (optional, but helpful)
            imports = "import cv2\nimport numpy as np\n\n# Assuming 'img' is your input image (e.g., read by cv2.imread)\n# --- Start of generated code ---\n"
            full_code = imports + generated_code + "\n# --- End of generated code ---"
            # Optional: assign result to a common variable name like 'processed_img' if not done in template
            # if 'processed_img =' not in generated_code:
            #     full_code += "\nprocessed_img = result # Assign the result if needed"

            self.code_text.config(state=tk.NORMAL)
            self.code_text.delete(1.0, tk.END)
            self.code_text.insert(tk.END, full_code)
            self.code_text.config(state=tk.DISABLED)
            self.code_button.config(state=tk.NORMAL)

        except KeyError as e:
            error_msg = f"Code generation error: Missing parameter '{e}' in current values for template."  # 保持英文错误
            print(error_msg)
            self._clear_code_text(error_msg)
            self.code_button.config(state=tk.DISABLED)
        except Exception as e:
            error_msg = f"Code generation error: {e}"  # 保持英文错误
            print(error_msg)
            self._clear_code_text(error_msg)
            self.code_button.config(state=tk.DISABLED)

    def _clear_code_text(self, message="Select a function to generate code."):  # 提示信息保持英文
        self.code_text.config(state=tk.NORMAL)
        self.code_text.delete(1.0, tk.END)
        self.code_text.insert(tk.END, message)
        self.code_text.config(state=tk.DISABLED)
        self.code_button.config(state=tk.DISABLED)

    def _copy_code_to_clipboard(self):
        code = self.code_text.get(1.0, tk.END).strip()
        if code:
            try:
                pyperclip.copy(code)
                print("Code copied to clipboard.")  # 保持英文日志
                # Optional: Show a brief confirmation message?
                # messagebox.showinfo("成功", "代码已复制到剪贴板") # Example message box (UI translation)
            except Exception as e:
                print(f"Failed to copy code to clipboard: {e}")  # 保持英文日志
                messagebox.showwarning("警告",
                                       f"无法复制到剪贴板: {e}\n请确保 'pyperclip' 模块已安装并且可用。")  # UI: 消息框中文

    def _save_image(self):
        if self.processed_img_cv is None:
            messagebox.showwarning("警告", "没有可保存的处理后图像。")  # UI: 消息框中文
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            cv2.imwrite(file_path, self.processed_img_cv)
            print(f"Processed image saved to {file_path}")  # 保持英文日志
            messagebox.showinfo("成功", f"图像已保存到:\n{file_path}")  # UI: 消息框中文
        except Exception as e:
            print(f"Error saving image: {e}")  # 保持英文日志
            messagebox.showerror("错误", f"保存图像失败: {e}")  # UI: 消息框中文

    def _save_params(self):
        if not self.selected_func_name or not self.current_params:
            messagebox.showwarning("警告", "没有可保存的参数。请先选择一个功能。")  # UI: 消息框中文
            return

        params_to_save = {
            "function_name": self.selected_func_name,
            "parameters": self.current_params
        }

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Parameter Files", "*.json"), ("All files", "*.*")],
            title="保存参数文件"  # UI: 对话框标题中文
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, indent=4)
            print(f"Parameters saved to {file_path}")  # 保持英文日志
            messagebox.showinfo("成功", f"参数已保存到:\n{file_path}")  # UI: 消息框中文
        except Exception as e:
            print(f"Error saving parameters: {e}")  # 保持英文日志
            messagebox.showerror("错误", f"保存参数失败: {e}")  # UI: 消息框中文

    def _load_params(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Parameter Files", "*.json"), ("All files", "*.*")],
            title="加载参数文件"  # UI: 对话框标题中文
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            func_name = loaded_data.get("function_name")
            params = loaded_data.get("parameters")

            if not func_name or not params:
                raise ValueError("JSON 文件格式无效或缺少必要数据。")  # 错误信息可考虑翻译

            # Check if the loaded function exists in our list
            if func_name not in OPENCV_FUNCTIONS:
                raise ValueError(f"加载的函数 '{func_name}' 当前应用不支持。")  # 错误信息可考虑翻译

            # Select the function in the combobox
            self.func_combobox.set(func_name)
            self._on_function_select()  # This rebuilds the UI for the function

            # Update parameter widgets/variables with loaded values
            func_info = OPENCV_FUNCTIONS.get(func_name)
            params_def = func_info.get("params", {})

            loaded_count = 0
            for name, value in params.items():
                if name in self.param_vars:
                    var = self.param_vars[name]
                    config = params_def.get(name, {})
                    param_type = config.get("type")

                    try:
                        # Attempt to set value, handling type consistency
                        if param_type == "slider" and isinstance(var, tk.IntVar):
                            var.set(int(value))
                        elif param_type == "float_slider" and isinstance(var, tk.DoubleVar):
                            var.set(round(float(value), 4))  # Round loaded float for display
                        elif param_type == "dropdown" and isinstance(var, tk.StringVar):
                            # Check if the loaded value is valid for the dropdown
                            if value in config.get("options", []):
                                var.set(str(value))
                            else:
                                print(
                                    f"Warning: Loaded value '{value}' for dropdown '{name}' is not a valid option. Using default.")  # 保持英文日志
                                # Optionally set to default or keep current var value
                        elif param_type == "checkbox" and isinstance(var, tk.BooleanVar):
                            var.set(bool(value))
                        else:
                            print(
                                f"Warning: Type mismatch or unknown type for param '{name}'. Skipping load for this param.")  # 保持英文日志
                            continue  # Skip this parameter

                        # Update internal state and potentially the slider position via _on_entry_change or _on_param_change
                        # Easiest is to just call _on_param_change after setting the var
                        # Need the correct type hint for _on_param_change
                        type_hint_map = {"slider": "int", "float_slider": "float", "dropdown": "str",
                                         "checkbox": "bool"}
                        hint = type_hint_map.get(param_type)
                        if hint:
                            # We need the *actual* value set in the var, not just 'value' from file
                            current_var_value = var.get()
                            # Manually update current_params before calling apply_processing
                            self.current_params[name] = current_var_value
                            # If slider, update its visual position too (IntVar/DoubleVar already linked)
                            # self._on_param_change(name, current_var_value, hint) # This might trigger redundant processing
                            loaded_count += 1

                    except Exception as set_err:
                        print(f"Warning: Could not set parameter '{name}' to value '{value}': {set_err}")  # 保持英文日志

            print(f"Parameters loaded from {file_path}. {loaded_count} parameters applied.")  # 保持英文日志

            # Apply processing with the newly loaded parameters
            # Update internal current_params directly before applying
            self.current_params.update({k: v for k, v in params.items() if
                                        k in self.current_params})  # Update only existing keys safely? Or trust the loop above?
            # The loop above should have updated self.current_params correctly via _on_param_change calls indirectly or direct assignment

            if self.original_img_cv is not None:
                self._apply_processing(update_params=False)  # Params should be up-to-date from loop
                self._generate_code()


        except json.JSONDecodeError:
            messagebox.showerror("错误",
                                 f"加载参数失败: 文件 '{os.path.basename(file_path)}' 不是有效的 JSON 文件。")  # UI: 消息框中文
        except ValueError as e:
            messagebox.showerror("错误", f"加载参数失败: {e}")  # UI: 消息框中文
        except Exception as e:
            print(f"Error loading parameters: {e}")  # 保持英文日志
            messagebox.showerror("错误", f"加载参数时发生未知错误: {e}")  # UI: 消息框中文

    def _on_closing(self):
        # Release camera resource if active
        if self.camera_active and self.cap:
            self.cap.release()
            print("Camera released on closing.")  # 保持英文日志
        self.destroy()  # Close the Tkinter window


# --- 主程序入口 ---
if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()
