{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "35HaIJ7yuExg"
      },
      "outputs": [],
      "source": [
        "# import shutil\n",
        "# shutil.rmtree(\"/content/sample_data\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import zipfile\n",
        "file = zipfile.ZipFile(\"/content/xray.yolov11.zip\")\n",
        "file.extractall()"
      ],
      "metadata": {
        "id": "zY2IaBTSBG4z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install ultralytics"
      ],
      "metadata": {
        "collapsed": true,
        "id": "jOfKYYMqJBN_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "843bb0dd-59f1-486e-ddad-782c7b4a523a"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: ultralytics in /usr/local/lib/python3.12/dist-packages (8.4.112)\n",
            "Requirement already satisfied: filelock>=3.16.1 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (3.29.7)\n",
            "Requirement already satisfied: numpy>=1.23.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (2.0.2)\n",
            "Requirement already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (3.10.0)\n",
            "Requirement already satisfied: opencv-python!=4.13.0.90,>=4.6.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (5.0.0.93)\n",
            "Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (11.3.0)\n",
            "Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (6.0.3)\n",
            "Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (2.32.4)\n",
            "Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (2.11.0+cpu)\n",
            "Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (0.26.0+cpu)\n",
            "Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (5.9.5)\n",
            "Requirement already satisfied: polars>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (1.35.2)\n",
            "Requirement already satisfied: nvidia-ml-py>=12.0.0 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (13.610.43)\n",
            "Requirement already satisfied: ultralytics-thop>=2.1.2 in /usr/local/lib/python3.12/dist-packages (from ultralytics) (2.1.5)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.3.3)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (4.63.0)\n",
            "Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.5.0)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (26.2)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (3.3.2)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.3.0->ultralytics) (2.9.0.post0)\n",
            "Requirement already satisfied: polars-runtime-32==1.35.2 in /usr/local/lib/python3.12/dist-packages (from polars>=0.20.0->ultralytics) (1.35.2)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests>=2.23.0->ultralytics) (3.4.9)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests>=2.23.0->ultralytics) (3.18)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests>=2.23.0->ultralytics) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests>=2.23.0->ultralytics) (2026.6.17)\n",
            "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (4.16.0)\n",
            "Requirement already satisfied: setuptools<82 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (75.2.0)\n",
            "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (1.14.0)\n",
            "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (3.6.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (3.1.6)\n",
            "Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.8.0->ultralytics) (2025.3.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.17.0)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.8.0->ultralytics) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.8.0->ultralytics) (3.0.3)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from ultralytics import YOLO\n",
        "\n",
        "model = YOLO(\"yolo11n.pt\")"
      ],
      "metadata": {
        "collapsed": true,
        "id": "xzQGkdL0PzGU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "23ea2ef7-e1d9-47c6-91f7-fe981f24ce32"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\r\u001b[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt to 'yolo11n.pt': 100% ━━━━━━━━━━━━ 5.4MB 56.9MB/s 0.1s\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "results = model.train(\n",
        "    data=\"/content/data.yaml\",\n",
        "    epochs=10,\n",
        "    imgsz=352,\n",
        "    batch=16,\n",
        "    workers=2\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Da5lRIZ6TApR",
        "outputId": "784308de-dc53-4658-be93-e01c8bf2a6ae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Ultralytics 8.4.112 🚀 Python-3.12.13 torch-2.11.0+cpu CPU (Intel Xeon CPU @ 2.20GHz)\n",
            "\u001b[34m\u001b[1mengine/trainer: \u001b[0magnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=False, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/data.yaml, degrees=0.0, deterministic=True, device=, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=10, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=352, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=train-3, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/runs/detect/train-3, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=2, workspace=None\n",
            "Overriding model.yaml nc=80 with nc=1\n",
            "\n",
            "                   from  n    params  module                                       arguments                     \n",
            "  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 \n",
            "  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                \n",
            "  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      \n",
            "  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            "  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     \n",
            "  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            "  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           \n",
            "  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              \n",
            "  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           \n",
            "  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 \n",
            " 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 \n",
            " 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          \n",
            " 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
            " 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           \n",
            " 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
            " 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          \n",
            " 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
            " 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
            " 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           \n",
            " 23        [16, 19, 22]  1    430867  ultralytics.nn.modules.head.Detect           [1, 16, None, [64, 128, 256]] \n",
            "YOLO11n summary: 182 layers, 2,590,035 parameters, 2,590,019 gradients, 6.4 GFLOPs\n",
            "\n",
            "Transferred 448/499 items from pretrained weights\n",
            "Freezing layer 'model.23.dfl.conv.weight'\n",
            "WARNING ⚠️ \u001b[34m\u001b[1mtrain: \u001b[0mSlow image access detected (ping: 0.0±0.0 ms, read: 6.9±1.5 MB/s, size: 10.9 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/\n",
            "\u001b[K\u001b[34m\u001b[1mtrain: \u001b[0mScanning /content/train/labels.cache... 782 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 782/782 105.8Mit/s 0.0s\n",
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))\n",
            "WARNING ⚠️ \u001b[34m\u001b[1mval: \u001b[0mSlow image access detected (ping: 0.0±0.0 ms, read: 18.7±25.9 MB/s, size: 11.5 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/\n",
            "\u001b[K\u001b[34m\u001b[1mval: \u001b[0mScanning /content/valid/labels.cache... 97 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 97/97 25.4Mit/s 0.0s\n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... \n",
            "\u001b[34m\u001b[1moptimizer:\u001b[0m AdamW(lr=0.002, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)\n",
            "Plotting labels to /content/runs/detect/train-3/labels.jpg... \n",
            "Image sizes 352 train, 352 val\n",
            "Using 0 dataloader workers\n",
            "Logging results to \u001b[1m/content/runs/detect/train-3\u001b[0m\n",
            "Starting training for 10 epochs...\n",
            "Closing dataloader mosaic\n",
            "\u001b[34m\u001b[1malbumentations: \u001b[0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       1/10         0G      2.848       3.96      2.159         19        352: 100% ━━━━━━━━━━━━ 49/49 5.1s/it 4:09\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 3.4s/it 13.7s\n",
            "                   all         97        114    0.00216      0.553    0.00199   0.000492\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       2/10         0G      2.618      3.253      1.937         20        352: 100% ━━━━━━━━━━━━ 49/49 4.2s/it 3:26\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.9s/it 11.7s\n",
            "                   all         97        114    0.00258      0.658     0.0218    0.00446\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       3/10         0G      2.518      2.984       1.94         14        352: 100% ━━━━━━━━━━━━ 49/49 4.1s/it 3:20\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 3.1s/it 12.5s\n",
            "                   all         97        114     0.0677      0.105     0.0279    0.00818\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       4/10         0G      2.475      2.666      1.887         16        352: 100% ━━━━━━━━━━━━ 49/49 4.1s/it 3:20\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.8s/it 11.2s\n",
            "                   all         97        114      0.163      0.188      0.117     0.0438\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       5/10         0G      2.368      2.456      1.807         17        352: 100% ━━━━━━━━━━━━ 49/49 4.0s/it 3:16\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.7s/it 11.0s\n",
            "                   all         97        114      0.177      0.246      0.152     0.0614\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       6/10         0G      2.311      2.274      1.802         19        352: 100% ━━━━━━━━━━━━ 49/49 4.0s/it 3:16\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.4s/it 9.5s\n",
            "                   all         97        114      0.401      0.289      0.305      0.125\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       7/10         0G      2.267      2.125      1.758         16        352: 100% ━━━━━━━━━━━━ 49/49 4.3s/it 3:31\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.8s/it 11.1s\n",
            "                   all         97        114       0.54      0.325      0.356       0.15\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       8/10         0G      2.197      1.974      1.715         16        352: 100% ━━━━━━━━━━━━ 49/49 4.2s/it 3:27\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.5s/it 10.0s\n",
            "                   all         97        114      0.428      0.394      0.387      0.146\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K       9/10         0G      2.132      1.859      1.664         16        352: 100% ━━━━━━━━━━━━ 49/49 4.0s/it 3:17\n",
            "\u001b[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.5s/it 10.0s\n",
            "                   all         97        114      0.544      0.447      0.446      0.172\n",
            "\n",
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
            "\u001b[K      10/10         0G      2.117      1.803      1.675         22        352: 44% ━━━━━─────── 22/49 4.1s/it 1:34<1:51"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "new_model = YOLO(\"/content/runs/detect/train-3/weights/best.pt\")"
      ],
      "metadata": {
        "id": "0bgvvUeFVZQi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "result = new_model(\"/content/test/images/all_0_157_png_jpg.rf.0JSuCQJHI4gGFWzDsK5t.jpg\", save=True)"
      ],
      "metadata": {
        "id": "w4-SKLO0WKtM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "result[0].plot()"
      ],
      "metadata": {
        "id": "5KEtCqCJWb01"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install gradio as gr"
      ],
      "metadata": {
        "id": "vEtA_eKcYCJU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def gr_function(image):\n",
        "  result = new_model(\"/content/runs/detect/train-3/weights/best.pt\")\n",
        "  image = result[0].plot()\n",
        "  return image\n",
        ""
      ],
      "metadata": {
        "id": "HsQMlKM1YJHe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "app = gr.Interface(\n",
        "    fn=gr_function,\n",
        "    inputs=gr.Image(type=\"numpy\"),\n",
        "    outputs=gr.Image(type=\"numpy\")\n",
        ")"
      ],
      "metadata": {
        "id": "CFZipwSWYpwH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "app.launch(share=True)"
      ],
      "metadata": {
        "id": "TUVSJC1DeMmp"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}