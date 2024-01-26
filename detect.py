# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()  #  __file__该代码的路径[str类型];path.resolve() 方法会把一个路径或路径片段的序列解析为一个绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:  # 模块的查询路径的列表
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换成相对路径

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_bbox
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)：模型的权重参数
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold：置信度的值
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 判断传入的路径是否是一个文件地址的后缀类型是否在已给格式之中
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断给的地址是否是网络流地址
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)  # 判断打开的路径是不是电脑摄像头的路径
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories：新建一个保存结果的文件夹
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run：创建一个保存结果的文件夹
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # Load model:加载模型权重
    device = select_device(device)  # 选择加载模型的设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)  # 加载对应学习框架的模型权重
    stride, names, pth, jit, onnx, engine = model.stride, model.names, model.pth, model.jit, model.onnx, model.engine  # 模型相关的参数值;pt表示是否是pytroch的模型
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader：加载待预测的图片
    if webcam:  # 路径是不是电脑摄像头的路径
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference：执行模型推理过程
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup：GPU热身环节？  #  模型开始能用了吗？？？？？？？？！！！！
    dt, seen = [0.0, 0.0, 0.0], 0  # 两个变量用来储存结果信息[储存什么信息？？？]

    count = 0  # 打印测试
    for path, im, im0s, vid_cap, s in dataset:
       #  path：检测图片的路径
       #  im：shape为[3,640,480]
       #  im0s：
       #  vid_cap：
       #  s：
       #  dataset：
       #  图片做预处理
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)  # torch.Size([3,640,480])
        im = im.half() if half else im.float()  # uint8 to fp16/32:判断模型是否用到半精度
        im /= 255  # 0 - 255 to 0.0 - 1.0：该行操作是归一化操作
        if len(im.shape) == 3:  # 判断一下数据的维度是否是四维的包含batch这一维度
            im = im[None]  # expand for batch dim, torch.Size([1,3,640,640])
        t2 = time_sync()
        dt[0] += t2 - t1  # 图片预处理所用的时间

        # Inference：对经过处理后的图片进行预测
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # visualize为False 暂时还没看是干嘛的！！
        # model为DetectMultiBackend，这里model()是DetectMultiBackend的前向传播的过程
        # pred: 网络的输出结果
        pred = model(im, augment=augment, visualize=visualize)  # torch.Size([1,18900,85]):18900是框的个数；visualize：模型推断的过程中是否保存特征图；
        t3 = time_sync()
        dt[1] += t3 - t2  # 图片进行预测所用的时间
        # NMS
        """
        pred: 网络的输出结果
        conf_thres:置信度阈值
        ou_thres:iou阈值
        classes: 是否只保留特定的类别
        agnostic_nms: 进行nms是否也去除不同类别之间的框  ？？？ 不太清楚是干嘛的？？？
        max-det: 保留的最大检测框数量
        ---NMS, 预测框格式: xywh(中心点+长宽)-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor], 长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
        """
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 1,5,6： 5指的是18900个目标降低到5个目标
        # 6指的是框的左上角xy右下角xy 置信度 种类的概率
        dt[2] += time_sync() - t3  # NMS所用的时间

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions：所有的检测框画到原图中并且保存结果

        for i, det in enumerate(pred):  # per image： pred指的是一个batch的所有图片
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg：图片的保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string：图片的尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh：获得原图的宽高的大小
            imc = im0.copy() if save_crop else im0  # for save_crop：检测框是否裁剪下来
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 好像是框绘图相关的
            if len(det):  # 是否有框
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # 坐标映射

                # Print results
                for c in det[:, -1].unique():  # 遍历每个框
                    n = (det[:, -1] == c).sum()  # detections per class  每张图片检测框的个数
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    count += int(n.item())
                    # print('---------这张图片检测出来框的个数:', n.item())  # 这张图片检测出来框的个数

                # Write results
                for *xyxy, conf, cls in reversed(det):  # 是否保存预测结果
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf*100:.2f}')  # 控制是否有类别标签和置信度
                        annotator.box_label(xyxy, label, color=colors(c, True))  # xyxy是list类型，前两个元素是目标的左上点的坐标，后面两个元素是目标的后下点两个坐标
                        # print('-------------------: ', xyxy,type(xyxy),int(len(xyxy)/4))  # -------------自己测试的是不是图像的框
                        #  打印测试结果：[tensor(630., device='cuda:0'), tensor(648., device='cuda:0'), tensor(710., device='cuda:0'), tensor(720., device='cuda:0')]
                        if save_crop:  # 是否保存截下来的目标框
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            im0 = annotator.result()  # 返回画好的图片
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # print('-----------------------------检测所有图的检测框数量：', count)  # 打印测试

    # Print results：打印输出信息
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # 定义一些命令行可以穿入的参数：
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')

    parser.add_argument('--weights', nargs='+', type=str, default=r'/data/wangzerui/exp567/weights/best.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default=r'/data/wangzerui/23_3_23_thin_thick_split/val/images', help='file/dir/URL/glob, 0 for webcam')  # 要检测的对象的路径，这个路径下的内容必须是图片或者是视频
    parser.add_argument('--data', type=str, default=r"/home/wangzerui/code_all/to101_yolov5-6.1/data/flame.yaml", help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')  # max_det：表示一张图中最大能检测多少个目标
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=r'/data/wangzerui/ours_vis/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')  # 当将default=3中的3改小，整个检测框和字体大小会整体对应变小，若改大，则整个检测框和字体大小会整体对应变大
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    # 使得输入的imgsz[x]或[x,x]转变为输入的imgsz为[x,x]
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 参数信息打印：
    print_args(FILE.stem, opt)  # opt是用来存储所有的参数信息
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))  # 检测requirements文件下的包是否成功安装
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()  # 解析负责传入的参数
    main(opt)  # opt的参数信息传入给main函数
