import cv2
import numpy as np


def preprocess_image(image):
    # Convert the image to grayscale if it's not already in grayscale
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    return gray_image


def find_largest_contour(image, num_largest=1):
    # Find edges in the image using Canny edge detection

    edges = cv2.Canny(image, 150, 200, L2gradient=1)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]

    # 获取最大的num_largest个轮廓的索引
    largest_indices = np.argsort(areas)[-num_largest:]

    # 提取最大的num_largest个轮廓
    largest_contours = [contours[i] for i in largest_indices]

    return largest_contours


def find_min_area_rect(image, contour):
    # Draw the contour on a blank image to use in cv2.minAreaRect
    blank_image = np.zeros_like(image)
    cv2.drawContours(blank_image, [contour], 0, 255, thickness=cv2.FILLED)

    # Find the minimum area rectangle enclosing the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def rotate_and_resize(image, box, target_size):
    # Sort the box points in clockwise order
    box = np.array(box)
    ordered_box = np.array([box[np.argmin(box[:, 0])], box[np.argmax(box[:, 0])],
                            box[np.argmax(box[:, 1])], box[np.argmin(box[:, 1])]], dtype=np.float32)

    # Calculate the perspective transform matrix
    target_points = np.array([[0, 0], [target_size[0], 0], target_size, [0, target_size[1]]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered_box, target_points)

    # Apply the perspective transform to obtain the rotated and resized image
    rotated_resized_image = cv2.warpPerspective(image, M, target_size)
    return rotated_resized_image


def bit_color_merge(image, lower_hsv, upper_hsv):
    # 转换到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建颜色掩码
    mask_color = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 腐蚀和膨胀
    kernel = np.ones((5, 5), np.uint8)
    mask_color = cv2.erode(mask_color, kernel, iterations=1)
    mask_color = cv2.dilate(mask_color, kernel, iterations=1)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 获取最大轮廓的边界框
    x, y, w, h = cv2.boundingRect(max_contour)

    # 创建一个全黑的mask
    mask_rect = np.zeros_like(mask_color)

    # 将边界框区域置为白色
    mask_rect[y:y + h, x:x + w] = 255

    # 将原始图像复制一份并转为灰度
    image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # 使用大津法进行阈值化
    _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 与之前的mask进行与运算
    result = cv2.bitwise_and(thresh, mask_rect)

    return result


def template_matching(image, template_list, name_list=None):
    # 采用最小包围矩形的方式进行模板匹配
    result_image = image.copy()
    processed_image = preprocess_image(image)

    # Find the largest contour in the image
    c_res = find_largest_contour(processed_image)
    if len(c_res) >= 1:
        largest_contours = find_largest_contour(processed_image)
    else:
        return None, image, image
    result_image_list, rotated_resized_image_list, template_name_list = [], [], []
    # Draw the bounding box on the result image
    for largest_contour in largest_contours:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(result_image, [box], 0, (0, 255, 0), 2)
        # Rotate and resize the image to match the template size
        width, height = abs(int(rect[1][0])), abs(int(rect[1][1]))
        target_size = template_list[0].shape[::-1]
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        # 对图像进行缩放和旋转以矫正至模板大小

        corrected_img = cv2.resize(warped, target_size)
        rotated_resized_image = corrected_img

        # Loop through each template and perform template matching
        for i, template in enumerate(template_list):
            # Perform template matching
            match_result = cv2.matchTemplate(rotated_resized_image, template, cv2.TM_CCOEFF_NORMED)

            # Find the location of the best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            # If the match is good (similarity > 0.8), draw the bounding box and return the template name
            if max_val > 0.8:
                template_name = f"Template_{i + 1}" if name_list is None else name_list[i]
                result_image_list.append(result_image)
                rotated_resized_image_list.append(rotated_resized_image)
                template_name_list.append(template_name)
            else:
                result_image_list.append(result_image)
                rotated_resized_image_list.append(rotated_resized_image)
                template_name_list.append(None)
    return template_name_list, result_image_list, rotated_resized_image_list


def binarize_image(gray_image, threshold_value):
    # 使用OpenCV阈值化函数，设置最大值为255（白色）
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_image


def make_template(template_class_nums, template_class_names, cap_num=0):
    """
    根据template_class_nums的数量对应template_class_names生成模板
    :param template_class_nums:
    :param template_class_names:
    :param cap_num:
    :return:
    """
    import time
    cap = cv2.VideoCapture(cap_num)
    template_class_list = []
    templates = [cv2.imread(r"C:\Users\Mili\PycharmProjects\opencv_utils\default_t.jpg",
                            cv2.IMREAD_GRAYSCALE)]
    for class_index in range(template_class_nums):
        for route_index in range(4):
            re, image = cap.read()
            if re:
                image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
                ret, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if ret:
                    print(f"请拍摄{90 * route_index}°旋转的照片{template_class_names[class_index]}")
                    t = time.time()
                    while time.time() - t <= 8:
                        re, image = cap.read()
                        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
                        ret, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        template_name_list, yuanshi_img_list, rotated_resized_image_list = template_matching(image_gray,
                                                                                                             templates)
                        if rotated_resized_image_list is not None and rotated_resized_image_list[0] is not None:
                            cv2.imshow("res", rotated_resized_image_list[0])
                            cv2.waitKey(1)

                    res = input("拍完之后按任意键进入下一步\n")
                    time.sleep(5)
                    if len(res) >= 1:
                        cv2.imwrite(
                            r"\templetes" + "\\" + f"{template_class_names[class_index]}_{90 * route_index}.jpg",
                            image_gray)
                        print("已经保存至",
                              r"\templetes" + "\\" + f"{template_class_names[class_index]}_{90 * route_index}.jpg")
                    template_class_list.append(template_class_names[class_index])
    return template_class_list


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    template1 = cv2.imread(r"C:\Users\Mili\PycharmProjects\opencv_utils\templetes_char\original.jpg",
                           cv2.IMREAD_GRAYSCALE)
    template2 = cv2.imread(r"C:\Users\Mili\PycharmProjects\opencv_utils\templetes_char\rotate_90.jpg",
                           cv2.IMREAD_GRAYSCALE)
    template3 = cv2.imread(r"C:\Users\Mili\PycharmProjects\opencv_utils\templetes_char\rotate_180.jpg",
                           cv2.IMREAD_GRAYSCALE)
    template4 = cv2.imread(r"C:\Users\Mili\PycharmProjects\opencv_utils\templetes_char\rotate_270.jpg",
                           cv2.IMREAD_GRAYSCALE) #读取文件
    templates = [template1]
    while True:
        ret, image = cap.read()
        # image_gray = bit_color_merge(image, np.array([0, 0, 186]), np.array([255, 255, 255]))
        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #大津法
        cv2.imshow(f"原始img", image_gray)
        cv2.waitKey(1)
        # image_gray = binarize_image(image_gray, threshold_value=130)
        # 在图像中查找多个模板并绘制矩形框
        template_name_list, yuanshi_img_list, rotated_resized_image_list = template_matching(image_gray, templates)
        for i, rotated_resized_image in enumerate(rotated_resized_image_list):

            if rotated_resized_image is not None and template_name_list is not None and template_name_list[
                i] is not None:
                print(template_name_list[i])
            cv2.imshow(f"原始img", yuanshi_img_list[i])
            cv2.waitKey(1)
            cv2.imshow(f"Match{i}", rotated_resized_image)


        # cv2.imwrite(r"C:\Users\Mili\PycharmProjects\opencv_utils\templetes_char\original.jpg", rotated_resized_image)
     