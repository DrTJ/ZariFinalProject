# راهنمای نصب و اجرای پروژه SSD Cascade + Deep SORT

این راهنما مرحله‌به‌مرحله نحوه نصب و اجرای پروژه `ssd_cascade_mot` را توضیح می‌دهد. این پروژه سه حالت تشخیص را با هم مقایسه می‌کند:

- **full (بدون کسکید):** مدل سنگین SSD300-VGG16 روی همه فریم‌ها
- **lite:** مدل سبک SSDLite320-MobileNetV3 روی همه فریم‌ها
- **cascade:** مدل سبک روی همه فریم‌ها، و مدل سنگین فقط وقتی لازم است

ردیابی با Deep SORT هم به‌صورت اختیاری اضافه می‌شود.

## ۱. پیش‌نیازها

- **سیستم‌عامل:** ویندوز ۱۰/۱۱، لینوکس یا macOS
- **پایتون:** نسخه ۳.۹ تا ۳.۱۲ (توصیه: ۳.۱۱)
  - دانلود: https://www.python.org/downloads/
  - در ویندوز هنگام نصب حتماً گزینه **Add Python to PATH** را تیک بزنید.
- **فضای دیسک:** حدود ۳ تا ۵ گیگابایت (برای PyTorch و مدل‌ها)
- **اینترنت:** برای نصب کتابخانه‌ها و دانلود خودکار وزن‌های مدل در اولین اجرا (حدود ۱۵۰ مگابایت)
- **کارت گرافیک (اختیاری):** کارت NVIDIA سرعت را چند برابر می‌کند، ولی پروژه روی CPU هم اجرا می‌شود.

برای بررسی نسخه پایتون، در ترمینال (یا Command Prompt در ویندوز) بنویسید:

```bash
python --version
```

## ۲. باز کردن پروژه

1. فایل `ssd_cascade_mot.zip` را از حالت فشرده خارج کنید.
2. ترمینال را باز کنید و وارد پوشه پروژه شوید:

```bash
cd ssd_cascade_mot
```

ساختار پوشه:

- `cascade_mot/` کد اصلی (تشخیص، کسکید، ردیابی، معیارها، اجرا)
- `tests/` تست‌های واحد
- `data/vtest.avi` ویدیوی نمونه عابران پیاده
- `results/` محل ذخیره خروجی‌ها
- `requirements.txt` فهرست کتابخانه‌ها

## ۳. ساخت محیط مجازی (Virtual Environment)

محیط مجازی باعث می‌شود کتابخانه‌های این پروژه با بقیه سیستم تداخل نداشته باشند.

**ویندوز:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**لینوکس / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

بعد از فعال‌سازی، عبارت `(.venv)` در ابتدای خط ترمینال دیده می‌شود. هر بار که ترمینال جدید باز می‌کنید، دوباره دستور فعال‌سازی را اجرا کنید.

سپس pip را به‌روز کنید:

```bash
python -m pip install --upgrade pip
```

## ۴. نصب PyTorch

**اگر کارت گرافیک NVIDIA دارید:** ابتدا درایور کارت را از https://www.nvidia.com/Download/index.aspx نصب کنید. سپس به https://pytorch.org/get-started/locally/ بروید، سیستم‌عامل، Pip، Python و نسخه CUDA را انتخاب کنید و دستوری را که سایت می‌دهد اجرا کنید. برای مثال:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**اگر فقط CPU دارید (یا مک):**

```bash
pip install torch torchvision
```

بررسی نصب:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

اگر خروجی `True` بود، کارت گرافیک شناسایی شده است.

## ۵. نصب بقیه کتابخانه‌ها

```bash
pip install -r requirements.txt
```

کتابخانه‌های اصلی:

- **opencv-python:** خواندن و نوشتن ویدیو — https://pypi.org/project/opencv-python/
- **deep-sort-realtime:** ردیاب Deep SORT — https://github.com/levan92/deep\_sort\_realtime
- **motmetrics:** محاسبه MOTA و IDF1 — https://github.com/cheind/py-motmetrics
- **matplotlib:** رسم نمودار مقایسه
- **pytest:** اجرای تست‌ها

## ۶. اجرای تست‌ها (اختیاری ولی توصیه‌شده)

```bash
pytest -q
```

باید پیام `13 passed` را ببینید. این تست‌ها بدون دانلود مدل اجرا می‌شوند و منطق کسکید و معیارها را بررسی می‌کنند.

## ۷. اولین اجرا: مقایسه سه حالت

```bash
python -m cascade_mot.run --video data/vtest.avi --classes person
```

- در اولین اجرا وزن‌های از پیش آموزش‌دیده روی COCO به‌طور خودکار از `download.pytorch.org` دانلود می‌شوند.
- اگر دانلود با خطا مواجه شد، اتصال اینترنت یا VPN/پروکسی خود را بررسی کنید. فایل‌ها در این مسیر ذخیره می‌شوند: ویندوز `C:\Users\<نام‌کاربری>\.cache\torch\hub\checkpoints` و لینوکس/مک `~/.cache/torch/hub/checkpoints`.
- لینک مستقیم وزن‌ها (برای دانلود دستی و قرار دادن در همان پوشه):
  - https://download.pytorch.org/models/ssdlite320\_mobilenet\_v3\_large\_coco-a79551df.pth
  - https://download.pytorch.org/models/ssd300\_vgg16\_coco-b556d3b4.pth

## ۸. اجرای کامل با ردیابی Deep SORT و ذخیره ویدیو

```bash
python -m cascade_mot.run --video data/vtest.avi --classes person --track --save-video
```

با کارت گرافیک:

```bash
python -m cascade_mot.run --video data/vtest.avi --classes person --track --save-video --device cuda --half
```

## ۹. خروجی‌ها (پوشه results)

- **summary.csv و summary.json:** یک ردیف برای هر حالت
- **comparison.png:** نمودار میله‌ای FPS، دقت نسبت به Full، و درصد استفاده از مدل سنگین
- **frames\_\<mode>.csv:** گزارش هر فریم (آیا Full استفاده شد، دلیل آن، تعداد تشخیص‌ها، زمان‌ها)
- **\<mode>.mp4:** ویدیوی برچسب‌دار (با `--save-video`)

معنی ستون‌های جدول:

- **fps:** تعداد فریم پردازش‌شده در ثانیه (بیشتر = سریع‌تر)
- **ms\_per\_frame:** زمان پردازش هر فریم به میلی‌ثانیه
- **full\_call\_rate:** درصد فریم‌هایی که مدل سنگین اجرا شد
- **recall\_vs\_full / precision\_vs\_full:** میزان تطابق تشخیص‌ها با حالت Full
- **speedup\_vs\_full:** چند برابر سریع‌تر از Full
- **unique\_ids / mean\_track\_len:** تعداد شناسه‌های ردیابی و طول متوسط مسیرها (شناسه کمتر و مسیر بلندتر یعنی تعویض شناسه کمتر)

## ۱۰. ارزیابی با داده واقعی MOT17 (برای پایان‌نامه)

1. داده MOT17 را از https://motchallenge.net/data/MOT17/ دانلود کنید (ثبت‌نام رایگان لازم است).
2. فریم‌های یک دنباله (مثلاً `MOT17-04-FRCNN/img1`) را با ffmpeg به ویدیو تبدیل کنید (دانلود ffmpeg: https://ffmpeg.org/download.html):

```bash
ffmpeg -framerate 30 -i MOT17-04-FRCNN/img1/%06d.jpg -c:v libx264 MOT17-04.mp4
```

3. اجرا با فایل برچسب واقعی:

```bash
python -m cascade_mot.run --video MOT17-04.mp4 --classes person --track --gt MOT17-04-FRCNN/gt/gt.txt --resize-width 960
```

در این حالت معیارهای استاندارد **MOTA**، **IDF1** و **تعداد تعویض شناسه (ID switches)** هم گزارش می‌شوند.

## ۱۱. تنظیم پارامترهای کسکید

مدل سنگین در چهار حالت فراخوانی می‌شود: فریم اول (first)، وجود تشخیص نامطمئن (uncertain)، تغییر تعداد اشیاء (count\_change)، و فریم کلیدی دوره‌ای (keyframe). خط `Cascade triggers` در خروجی نشان می‌دهد کدام دلیل بیشتر رخ داده است.

- **اگر مدل سنگین زیاد فراخوانی می‌شود:** `--min-uncertain 2` یا `--count-tolerance 1` بگذارید، یا بازه را باریک‌تر کنید: `--low 0.35 --high 0.55`
- **اگر دقت نسبت به Full پایین است:** بازه را بازتر کنید یا `--refresh-interval` را کمتر کنید (مثلاً ۱۵).
- **ویدیوی با وضوح بالا:** `--resize-width 960` سرعت را بسیار بیشتر می‌کند.

سایر گزینه‌ها: `--modes full cascade` (اجرای چند حالت خاص)، `--max-frames 300` (تعداد فریم‌ها)، `--classes person car bus` (کلاس‌های COCO)، `--threads 4` (تعداد هسته CPU).

## ۱۲. رفع مشکلات رایج

- **'python' is not recognized:** پایتون به PATH اضافه نشده؛ دوباره نصب کنید و گزینه Add to PATH را تیک بزنید، یا از `py` به‌جای `python` استفاده کنید.
- **No module named cascade\_mot:** دستور را از داخل پوشه `ssd_cascade_mot` اجرا کنید.
- **خطای دانلود وزن‌ها:** طبق بخش ۷ فایل‌ها را دستی دانلود و در پوشه checkpoints قرار دهید.
- **CUDA out of memory:** گزینه `--half` یا `--resize-width 640` را اضافه کنید.
- **ویدیوی خروجی پخش نمی‌شود:** از پخش‌کننده VLC استفاده کنید: https://www.videolan.org/vlc/
- **اجرا روی CPU خیلی کند است:** این طبیعی است (SSD300 روی CPU کند است)؛ با `--max-frames 100` تست کنید یا از کارت گرافیک یا Google Colab (https://colab.research.google.com) استفاده کنید.

## ۱۳. اتصال به ایده پایان‌نامه

مدل‌های فعلی از پیش آموزش‌دیده عمومی روی COCO هستند. برای تطابق کامل با روش پیشنهادی پایان‌نامه (SSD-Lite با داده محدودتر و SSD-Full با داده جامع‌تر)، هر دو مدل را روی دیتاست‌های خود fine-tune کنید و وزن‌ها را در فایل `cascade_mot/detectors.py` بارگذاری کنید. راهنمای fine-tune در torchvision: https://pytorch.org/tutorials/intermediate/torchvision\_tutorial.html

## ۱۴. آموزش (Fine-tune) دو مدل SSD-Lite و SSD-Full روی داده خودتان

ایده پایان‌نامه این است که SSD-Lite روی داده محدودتر (کلاس‌های کمتر) و SSD-Full روی داده جامع‌تر آموزش ببیند. این بخش هر دو را گام‌به‌گام توضیح می‌دهد. روی CPU آموزش بسیار کند است؛ کارت گرافیک NVIDIA یا Google Colab (https://colab.research.google.com، منوی Runtime ← Change runtime type ← GPU) توصیه می‌شود.

### ۱۴-۱. انتخاب و دانلود داده

- **MOT17 (عابر پیاده، برای ردیابی):** https://motchallenge.net/data/MOT17/
- **VisDrone (اشیای کوچک از دید پهپاد، ۱۰ کلاس):** https://github.com/VisDrone/VisDrone-Dataset
- **Airborne Object Detection (دیتاست ذکرشده در پایان‌نامه):** https://registry.opendata.aws/airborne-object-tracking/
- **ساخت داده اختصاصی:** تصاویر را با ابزار رایگان CVAT (https://www.cvat.ai) یا Label Studio (https://labelstud.io) برچسب بزنید و خروجی را با فرمت **COCO 1.0** بگیرید.

پیشنهاد عملی برای دو مدل:

| مدل | داده | کلاس‌ها |
| --- | --- | --- |
| SSD-Lite | زیرمجموعه کوچک (مثلاً فقط عابر و خودرو) | ۲ تا ۳ کلاس |
| SSD-Full | کل داده (مثلاً همه کلاس‌های VisDrone) | همه کلاس‌ها |

### ۱۴-۲. ساختار پوشه داده

داده را به فرمت COCO و این ساختار درآورید (برای هر مدل یک پوشه جدا):

```text
datasets/lite/
  images/train/   images/val/
  annotations/train.json   annotations/val.json
datasets/full/
  images/train/   images/val/
  annotations/train.json   annotations/val.json
```

- حدود ۸۰٪ تصاویر برای train و ۲۰٪ برای val.
- شناسه کلاس‌ها (category\_id) در فایل JSON باید از ۱ شروع شود؛ عدد ۰ برای پس‌زمینه رزرو است.
- برای تبدیل فرمت VisDrone یا YOLO به COCO می‌توانید از ابزار https://github.com/Weichen-Huang/visdrone2coco یا کتابخانه `pip install globox` استفاده کنید.

### ۱۴-۳. نصب کتابخانه کمکی

```bash
pip install pycocotools
```

در ویندوز اگر نصب خطا داد: `pip install pycocotools-windows`

### ۱۴-۴. اسکریپت آموزش

فایلی به نام `train.py` در پوشه اصلی پروژه بسازید و این کد را در آن قرار دهید. این کد مدل از پیش آموزش‌دیده روی COCO را می‌گیرد، لایه دسته‌بندی (head) را برای تعداد کلاس‌های شما عوض می‌کند و آموزش می‌دهد:

```python
import argparse, torch, torchvision
from torchvision.datasets import CocoDetection
from torchvision.models.detection import ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
import torchvision.transforms.functional as F
from functools import partial
from torch import nn

def build(model_name, num_classes):
    if model_name == "ssdlite":
        m = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
        ch = det_utils.retrieve_out_channels(m.backbone, (320, 320))
        na = m.anchor_generator.num_anchors_per_location()
        norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        m.head.classification_head = SSDLiteClassificationHead(ch, na, num_classes, norm)
    else:
        m = ssd300_vgg16(weights="DEFAULT")
        ch = det_utils.retrieve_out_channels(m.backbone, (300, 300))
        na = m.anchor_generator.num_anchors_per_location()
        m.head.classification_head = SSDClassificationHead(ch, na, num_classes)
    return m

class Coco(CocoDetection):
    def __getitem__(self, i):
        img, anns = super().__getitem__(i)
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in anns if a["bbox"][2] > 1 and a["bbox"][3] > 1]
        labels = [a["category_id"] for a in anns if a["bbox"][2] > 1 and a["bbox"][3] > 1]
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(labels, dtype=torch.int64)}
        return F.to_tensor(img), target

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ssdlite", "ssd300"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num-classes", type=int, required=True, help="تعداد کلاس‌ها + ۱ برای پس‌زمینه")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(a.model, a.num_classes).to(dev)
    ds = Coco(f"{a.data}/images/train", f"{a.data}/annotations/train.json")
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=2,
                                     collate_fn=lambda b: tuple(zip(*b)))
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=4e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    for ep in range(a.epochs):
        model.train(); total = 0
        for imgs, tgts in dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        sched.step()
        print(f"epoch {ep + 1}/{a.epochs}  loss {total / len(dl):.3f}")
        torch.save(model.state_dict(), a.out)
    print("saved", a.out)

if __name__ == "__main__":
    main()
```

### ۱۴-۵. اجرای آموزش

مقدار `--num-classes` برابر تعداد کلاس‌ها به‌علاوه ۱ (پس‌زمینه) است.

```bash
# SSD-Lite با ۲ کلاس (مثلاً عابر و خودرو)
python train.py --model ssdlite --data datasets/lite --num-classes 3 --epochs 30 --out ssd_lite.pth

# SSD-Full با ۱۰ کلاس VisDrone
python train.py --model ssd300 --data datasets/full --num-classes 11 --epochs 40 --batch 8 --out ssd_full.pth
```

نکات:

- اگر خطای کمبود حافظه گرفتید، `--batch` را به ۴ یا ۲ کاهش دهید.
- عدد loss باید در طول epochها کم شود. اگر ثابت ماند، `--lr` را به 0.005 کاهش دهید.
- اگر loss برابر nan شد، `--lr` را کمتر کنید و جعبه‌های خراب (عرض یا ارتفاع صفر) را از داده حذف کنید.

### ۱۴-۶. بارگذاری مدل‌های آموزش‌دیده در پروژه

در فایل `cascade_mot/detectors.py`، در کلاس `SSDDetector` و بعد از خط ساخت مدل، وزن‌های خودتان را بارگذاری کنید. ساده‌ترین راه: تابع `build` را از `train.py` وارد کنید و جایگزین ساخت مدل کنید:

```python
from train import build
model = build("ssdlite", 3)          # یا build("ssd300", 11)
model.load_state_dict(torch.load("ssd_lite.pth", map_location="cpu"))
```

سپس فهرست نام کلاس‌ها (`COCO_CATEGORIES`) را با نام کلاس‌های خودتان عوض کنید، به ترتیب category\_id و با `"__background__"` در خانه اول.

**نکته مهم:** در حالت کسکید، هر دو مدل باید شماره‌گذاری کلاس یکسانی داشته باشند تا مقایسه درست باشد. یعنی کلاس‌های SSD-Lite باید زیرمجموعه کلاس‌های SSD-Full با همان شماره‌ها باشند. اگر شماره‌ها فرق دارند، خروجی SSD-Lite را قبل از ارسال به ردیاب به شماره‌های SSD-Full نگاشت کنید.

### ۱۴-۷. ارزیابی دقت مدل‌ها (mAP)

برای گزارش در پایان‌نامه، mAP هر مدل را روی داده val با کتابخانه torchmetrics حساب کنید:

```bash
pip install torchmetrics
```

راهنمای آن: https://lightning.ai/docs/torchmetrics/stable/detection/mean\_average\_precision.html

سپس مقایسه کامل را با مدل‌های جدید دوباره اجرا کنید (بخش ۸ و ۱۰) و جدول FPS، دقت نسبت به Full، و MOTA/IDF1 را در فصل چهارم گزارش کنید.

### ۱۴-۸. منابع بیشتر

- آموزش رسمی fine-tune در torchvision: https://pytorch.org/tutorials/intermediate/torchvision\_tutorial.html
- اسکریپت‌های مرجع آموزش SSD در torchvision: https://github.com/pytorch/vision/tree/main/references/detection
- مقاله اصلی SSD: https://arxiv.org/abs/1512.02325
- مقاله Deep SORT: https://arxiv.org/abs/1703.07402
