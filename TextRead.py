import argparse
import random

import cv2
import math
import numpy as np
import os
import sys
import progressbar


def getTextFeatures(txt, font_scale=1, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    font_color = (255, 255, 255)
    back_color = (0, 0, 0)
    features = {}
    for char in txt:
        size = cv2.getTextSize(char, font, font_scale, thickness)
        wm = np.zeros((size[0][1], size[0][0], 3))
        wm[:] = back_color
        cv2.putText(wm, char, (0, size[0][1]), font, font_scale, font_color, thickness, cv2.LINE_AA)
        wm, _, _ = cv2.split(wm)
        features[char] = np.array(wm >= [1], dtype='int')
        print(char, features[char])
    return features



def encodeByText(img_path, txt, alpha, width, height, out_path):
    img = cv2.imread(img_path, -1)
    h, w, t = img.shape

    #绘制文本到一个空的图片上
    font_color = (0, 0, 0)
    back_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    size = cv2.getTextSize(txt, font, font_scale, thickness)
    wm = np.zeros((size[0][1] + 20, size[0][0] + 20, t))  # 为了方便识别, 保证文本和画布之间有10的 Padding
    wm[:] = back_color
    cv2.putText(wm, txt, (10, size[0][1] + 10), font, font_scale, font_color, thickness, cv2.LINE_AA)

    if width > 0 and height > 0:
        out_img = np.zeros(img.shape)
        with progressbar.ProgressBar(redirect_stdout=False, max_value=math.ceil(w / width) * math.ceil(h / height)) as bar:
            count = 0
            # 对图片按指定大小切片后,分别添加水印后,再重新拼接回一整张图片
            for i in range(0, w, width):
                for j in range(0, h, height):
                    w1 = width
                    h1 = height
                    # 图片大小不是总能被指定的大小整除,这里将不够指定晓得图片不添加水印直接拼接到最终图片上
                    if i + width > w:
                        w1 = width - (i + width - w)
                    if j + height > h:
                        h1 = height - (j + height - h)
                    tmp = np.zeros((h1, w1, t))
                    for x in range(w1):
                        for y in range(h1):
                            tmp[y][x] = img[j + y][i + x]
                    if w1 != width or h1 != height:
                        out_tmp = tmp
                    else:
                        out_tmp = encodeImg(tmp, wm, alpha)
                    for a in range(w1):
                        for b in range(h1):
                            out_img[j + b][i + a] = out_tmp[b][a]
                    bar.update(count)
                    count += 1
    else:
        out_img = encodeImg(img, wm, alpha)
    # 为了减少因图片压缩导致的水印数据丢失,始终保存为 png
    # if t == 4:
    if not out_path.endswith(".png"):
        out_path = out_path + ".png"
    cv2.imwrite(out_path, out_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    # else:
    #     if not out_path.endswith(".jpg"):
    #         out_path = out_path+".jpg"
    #     cv2.imwrite(out_path, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return out_path


def encodeImg(img, wm, alpha):
    h, w, t = img.shape
    wm_height, wm_width = wm.shape[0], wm.shape[1]
    x, y = list(range(int(h / 2))), list(range(int(w / 2)))
    random.seed(h + w)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(img.shape)
    for i in range(int(h / 2)):
        for j in range(int(w / 2)):
            if x[i] < wm_height and y[j] < wm_width:
                tmp[i][j] = wm[x[i]][y[j]]
                tmp[h - 1 - i][w - 1 - j] = tmp[i][j]
    img_f = np.fft.fft2(img)
    res_f = img_f + alpha * tmp
    res = np.fft.ifft2(res_f)
    res = np.real(res)
    return res


def getTextDatas(img):
    # 二值化
    closed = cv2.merge([img, img, img])
    gray = cv2.cvtColor(closed.astype('uint8'), cv2.COLOR_BGR2GRAY)
    _, closed = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    _, cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tmp = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(tmp) == 0:
        return None
    all_features = getTextFeatures('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    txt_arr = []
    for c in tmp:
        # 计算包含轮廓的最小矩形面积
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        crop_height = y2 - y1
        crop_width = x2 - x1
        # 裁剪为只包含文本的图片
        cropImg = img[y1:y1 + crop_height, x1:x1 + crop_width]
        # cv2.imshow("cropImg", cropImg)
        # cv2.waitKey(0)
        feature = np.array(cropImg >= [1], dtype='int')
        print(feature)
        max_num = 0
        max_char = None
        for f in all_features:
            num = np.mean(all_features[f] == feature)
            print(num)
            if num > max_num:
                max_num = num
                max_char = f
        txt_arr.append(max_char)
    return txt_arr






def textRead(img):
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # 读取灰度值
    # 由于解码出来的图片有很多深色杂点, 导致直接使用 cvtColor 转换成灰度图,杂点的灰度会高于文本,导致后面二值化后无法分析
    _, _, im_gray = cv2.split(img)
    # im_gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)  # 这个方法出来的图片背景很多白色杂点
    # 二值化
    _, out_tmp = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", out_tmp)
    # cv2.waitKey(0)
    # # 填充二维码中间黑色区域
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # closed = cv2.morphologyEx(out_tmp, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed", closed)
    # cv2.waitKey(0)
    # # 抹去文本周边可能存在的杂点
    # kernel = np.ones((2, 2), np.uint8)
    # closed = cv2.erode(out_tmp, kernel, iterations=4)
    # closed = cv2.dilate(closed, kernel, iterations=4)
    # cv2.imshow("closed", closed)
    # cv2.waitKey(0)
    # 不知道是不是前面没有使用cvtColor转换灰度图的原因,后面直接调用findContours会报错,所以这里先将图片合并成一个 BGR的图片,然后再转换成灰度度并再次二值化
    closed = cv2.merge([out_tmp, out_tmp, out_tmp])
    gray = cv2.cvtColor(closed.astype('uint8'), cv2.COLOR_BGR2GRAY)
    _, closed = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    _, cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tmp = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(tmp) == 0:
        return None
    c = tmp[0]
    # 计算包含轮廓的最小矩形面积(对于没有旋转过的图片,其实不需要这一步)
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    crop_height = y2 - y1
    crop_width = x2 - x1
    # 裁剪为只包含文本的图片
    cropImg = out_tmp[y1:y1 + crop_height, x1:x1 + crop_width]
    for x in range(crop_height):
        for y in range(crop_width):
            cropImg[x][y] = 255-cropImg[x][y]
    # cv2.imshow("cropImg", cropImg)
    # cv2.waitKey(0)
    # 抹去文本周边可能存在的杂点
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.erode(cropImg, kernel, iterations=1)
    # closed = cv2.dilate(closed, kernel, iterations=4)
    # cv2.imshow("closed", closed)
    # cv2.waitKey(0)
    return getTextDatas(closed)


def decode(ori_path, wm_path, alpha, width, height, step=1):
    ori = cv2.imread(ori_path, -1)
    vm = cv2.imread(wm_path, ori.shape[2])
    h, w, t = ori.shape
    if width > 0 and height > 0:
        ori_tmps = []
        for i in range(0, w, width):
            for j in range(0, h, height):
                w1 = width
                h1 = height
                # 图片大小不是总能被指定的大小整除,这里将不够指定晓得图片不添加水印直接拼接到最终图片上
                if i + width > w:
                    w1 = width - (i + width - w)
                if j + height > h:
                    h1 = height - (j + height - h)
                tmp = np.zeros((h1, w1, t))
                for x in range(w1):
                    for y in range(h1):
                        tmp[y][x] = ori[j + y][i + x]
                ori_tmps.append(tmp)

        with progressbar.ProgressBar(redirect_stdout=False,
                                     max_value=(int(width / 2 / step) * int(height / 2 / step)) * math.ceil(
                                          w / width) * math.ceil(h / height)) as bar:
            count = 0
            # 对待解码的图片按指定的大小裁剪,然后尝试解码并尝试识别其中的二维码,如果失败,则偏移一个像素后重试,最多重试 (width*height/4)*(ori.width/width)*(ori.height/height) 次
            for i in range(0, int(width / 2), step):
                for j in range(0, int(height / 2), step):
                    tmp = np.zeros((height, width, t))
                    for x in range(width):
                        for y in range(height):
                            if j + y >= len(vm) or i + x >= len(vm[j + y]):
                                break
                            tmp[y][x] = vm[j + y][i + x]
                    for out_tmp in decodeImg(ori_tmps, tmp, alpha):
                        if out_tmp is not None:
                            txt_data = textRead(out_tmp)
                            if txt_data is not None:
                                return txt_data

                        bar.update(count)
                        count += 1
    else:
        out_tmp = next(decodeImg([ori], vm, alpha))
        out_data = textRead(out_tmp)
        if out_data:
            return out_data
    return None


def decodeImg(ori_imgs, img, alpha):
    for ori_img in ori_imgs:
        if ori_img.shape[0] != img.shape[0] or ori_img.shape[1] != img.shape[1]:
            yield None
        else:
            h, w, t = ori_img.shape
            img_f = np.fft.fft2(img)
            ori_f = np.fft.fft2(ori_img)
            watermark = (img_f - ori_f) / alpha
            watermark = np.real(watermark)
            res = np.zeros(watermark.shape)
            x, y = list(range(int(h / 2))), list(range(int(w / 2)))
            random.seed(h + w)
            random.shuffle(x)
            random.shuffle(y)
            for i in range(int(h / 2)):
                for j in range(int(w / 2)):
                    res[x[i]][y[j]] = watermark[i][j]
                    # res[h - 1 - x[i]][w - 1 - y[j]] = res[x[i]][y[j]]
            # cv2.imwrite("dddd.jpg", res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            yield res
    return


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_error(msg):
    print('%s%s%s' % (Colors.FAIL, msg, Colors.ENDC))


def print_suc(msg):
    print('%s%s%s' % (Colors.OKGREEN, msg, Colors.ENDC))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(usage='%(prog)s <source_path> [options]')
    # parser.add_argument('source_path', help='原图片路径', nargs='?')
    # parser.add_argument('--text', '-t', help='指定需要写入的文本(编码时请提供!)', type=str, default=None)
    # parser.add_argument('--marked_path', '-m', help='指定需要解码的图片路径(解码操作请提供!)', type=str)
    # parser.add_argument('--out_path', '-o', help='加密后的图片存放路径,默认为原图片名加上"_encoded"', type=str, default=None)
    # parser.add_argument('--alpha', '-a', help='水印强度, 值越大,鲁棒性越强,但对原图片的影响越大. 默认值是 5', type=int, default=5)
    # parser.add_argument('--width', '-w', help='切片大小, 如果需要对原图切片并添加水印, 请指定该值. 如果指定该值,请确保指定的大小大于文字宽度的2倍,最好4倍以上.', type=int, default=0)
    # if len(sys.argv) < 2:
    #     print(parser.format_help())
    # else:
    #     args = parser.parse_args()
    #     original_path = os.path.join(os.getcwd(), args.source_path)
    #     if not os.path.exists(original_path):
    #         print_error('文件 %s 不存在!' % original_path)
    #         print(parser.format_help())
    #     elif args.text:
    #         output_path = args.out_path
    #         if output_path is None:
    #             fn, ext = os.path.splitext(original_path)
    #             output_path = '%s_encode' % fn
    #         encodeByText(original_path, args.text, args.alpha, args.width, args.width, output_path)
    #         print_suc('加密成功,加密后文件:%s' % output_path)
    #     elif args.marked_path:
    #         marked_path = os.path.join(os.getcwd(), args.marked_path)
    #         if not os.path.exists(marked_path):
    #             print_error('文件 %s 不存在!' % marked_path)
    #             print(parser.format_help())
    #         else:
    #             data = decode(original_path, marked_path, args.alpha, args.width, args.width)
    #             if data:
    #                 print_suc('解码成功,内容:%s' % data)
    #             else:
    #                 print_error('解码失败!')
    #     else:
    #         print_error('参数错误!')
    #         print(parser.format_help())

    path = "img"
    outPath = os.path.join(path, "out")
    file = "i.jpg"
    alpha1 = 1
    fn, ext = os.path.splitext(file)
    f = os.path.join(path, file)
    out_f = os.path.join(outPath, fn + "_out")
    de_f = os.path.join(outPath, fn + "_de")
    max_w = 0
    max_h = 0
    out_f = encodeByText(f, "fuck you", alpha1, max_w, max_h, out_f)
    # out_f = os.path.join(path, 'test2.png')
    decode(f, out_f, alpha1, max_w, max_h, 2)
    cv2.destroyAllWindows()
