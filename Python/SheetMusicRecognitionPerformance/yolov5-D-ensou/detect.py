# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch
import numpy as np
import time
import pyaudio
import wave as wave_module  # 'wave'モジュールを別名でインポート
from datetime import datetime  # 現在の日時の取得するためのモジュール

# 音階の周波数（ドレミファソラシド）
FREQUENCIES = {
    "do": 261.63,
    "re": 293.66,
    "mi": 329.63,
    "fa": 349.23,
    "so": 392.00,
    "ra": 440.00,
    "si": 493.88,
    "do_t": 523.25,  # 高いド
    "stop": 10.0
}

# サンプリングレート（1秒間に何回データを取るか）
SAMPLING_RATE = 44100

# 1つの音の長さ（秒）
DURATION = 0.5

# 無音の長さ（秒）
SILENCE_DURATION = 0.02

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / "yolov5s.pt",  # model path or triton URL(real time)
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    sound_xy_list = []
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # Write results
                sound_xy_list = []
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    # 音名、x座標、y座標の順にエクステンド
                    sound_xy_list.extend([label, xyxy[0].item(), xyxy[1].item()])
                    # print(sound_xy_list)

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
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
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return sound_xy_list


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    # リアルタイムカメラ発動！！
    # (run(**vars(opt)))

    # 保存先のディレクトリ
    save_dir = r"data\images"
    filename_list = []

    # images内の画像を削除
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # カメラの初期化
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けませんでした")
        exit()

    print("カメラが起動しました。's'キーを押して画像を保存し、'q'キーを押して終了します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした")
            break

        # フレームを表示
        cv2.imshow("Camera", frame)

        # キー入力を待つ
        key = cv2.waitKey(1) & 0xFF

        # 's'キーが押されたら画像を保存
        if key == ord('s'):
            filename = os.path.join(save_dir, f"image_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            filename_list.append(filename)
            print(f"画像を保存しました: {filename_list}")
        elif key == ord('q'):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

    return filename_list


# 近い値同士でグループ化
def near_group(arr, j):
    x_list = []  # グループ化リスト(3次元)
    now_x_list = [arr[0]]  # 現在のグループリスト(2次元)

    # 1から音の数だけループ
    for i in range(1, len(arr)):
        ans = arr[i][j] - arr[i - 1][j]

        # ソート後の隣り合う座標同士の差が40以下なら近い値として現在のグループリストに追加
        if ans <= 40:
            now_x_list.append(arr[i])
        # 違うならグループ化リストに追加し、現在のグループリストを初期化
        else:
            x_list.append(now_x_list)
            now_x_list = [arr[i]]
    x_list.append(now_x_list)
    return x_list


# リストの要素をs個ごとに分割し、二次元リストに
def split(arr, s):
    return [arr[i:i + s] for i in range(0, len(arr), s)]


# クイックソート(再帰関数)
def quicksort(arr, flag, i):
    # listの要素が1なら終了,flagが0なら昇順、flagが1なら降順
    if len(arr) <= 1:
        return arr
    elif flag == 0:
        median = arr[len(arr) // 2][i]
        left = [x for x in arr if x[i] < median]
        middle = [x for x in arr if x[i] == median]
        right = [x for x in arr if x[i] > median]
        return quicksort(left, 0, i) + middle + quicksort(right, 0, i)
    elif flag == 1:
        median = arr[len(arr) // 2][i]
        left = [x for x in arr if x[i] > median]
        middle = [x for x in arr if x[i] == median]
        right = [x for x in arr if x[i] < median]
        return quicksort(left, 1, i) + middle + quicksort(right, 1, i)


# サイン波を生成する関数
def make_sin_wave(frequency, duration, sampling_rate):
    # 0秒から指定した時間までの間を等間隔で分けた配列を作成
    t = np.linspace(0, duration, int(sampling_rate * duration), False)
    # 指定した周波数のサイン波を作成
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    return wave


# 音を再生する関数
def play_sound(note):
    # 音階の周波数を辞書(FREQUENCIES)から取得
    frequency = FREQUENCIES.get(note)
    if frequency:
        # 指定した音階のサイン波を生成
        wave = make_sin_wave(frequency, DURATION, SAMPLING_RATE)

        # PyAudioを使って音を再生する準備
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLING_RATE, output=True)  # ストリームを開く

        # サイン波データをバイト型に変換してストリームに書き込む（音を再生）
        stream.write(wave.astype(np.float32).tobytes())

        stream.stop_stream()  # ストリームを止める
        stream.close()  # ストリームを閉じる
        p.terminate()  # PyAudioを終了


# WAVファイルに音を保存する関数
def save_wave(filename, sound_data):
    # 波形データをWAVファイルとして保存する
    with wave_module.open(filename, 'w') as wf:
        wf.setnchannels(1)  # チャンネル数（1:モノラル2:ステレオ）
        wf.setsampwidth(2)  # サンプル幅（2bytes = 16bit）
        wf.setframerate(SAMPLING_RATE)  # サンプリングレート = 44100
        wf.writeframes((sound_data * 32767).astype(np.int16).tobytes())  # データを書き込む 2**15=32767をかけて16ビットの整数に変換


if __name__ == "__main__":
    weights = r"runs\train\outdir2\weights\best.pt"  # model path or triton URL(img)

    # リアルタイムカメラ準備！！
    opt = parse_opt()
    # 各画像のパス名を記録
    filename_list = main(opt)

    # 画像認識発動！！ 画像ごとに認識
    for filename in filename_list:
        sound_xy_list = []  # 音座標リスト(2次元→3次元)
        sound_result_list = []  # 音結果リスト(3次元)

        sound_xy_list.append(run(weights, filename))
        print(sound_xy_list)

        # 音ごとに[音名、x座標、y座標]で分割
        sound_xy_list = split(sound_xy_list[0], 3)
        print(sound_xy_list)

        # y座標で昇順にソート
        sound_xy_list = quicksort(sound_xy_list, 0, 2)
        print(sound_xy_list)

        # y座標の近い座標同士でグループ化
        sound_xy_list = near_group(sound_xy_list, 2)

        # グループごとにx座標で昇順にソート
        for sound_x_list in sound_xy_list:
            sound_result_list.append(quicksort(sound_x_list, 0, 1))
        print(sound_result_list)

        # すべての音を連結するためのリスト
        all_waves = []

        # 楽譜の行の数だけループ
        for sound_row_list in sound_result_list:
            # 行中の音の数だけループ
            for sound_list in sound_row_list:
                note = sound_list[0]
                print(note)

                # 音階の周波数を辞書(FREQUENCIES)から取得
                frequency = FREQUENCIES.get(note)
                if frequency:
                    # 指定した音階のサイン波を生成
                    wave = make_sin_wave(frequency, DURATION, SAMPLING_RATE)
                    # 生成したサイン波をリストに追加
                    all_waves.append(wave)

        # すべての音を1つの波形に連結
        full_wave = np.concatenate(all_waves)

        # 現在の日時を取得しファイル名にする
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # 音声ファイルの保存先パス
        save_path = fr"data\song\song_{timestamp}.wav"

        # WAVファイルとして保存
        save_wave(save_path, full_wave)

        # PyAudioを使って音を再生する準備
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLING_RATE, output=True)  # ストリームを開く

        # サイン波データをバイト型に変換してストリームに書き込む（音を再生）
        stream.write(full_wave.astype(np.float32).tobytes())

        stream.stop_stream()  # ストリームを止める
        stream.close()  # ストリームを閉じる
        p.terminate()  # PyAudioを終了
