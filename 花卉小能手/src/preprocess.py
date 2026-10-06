import sys
import random
import shutil
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

IMG_EXTS = {".jpg"}

def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()                
        with Image.open(path) as img:
            img.load()
            if img.size[0] == 0 or img.size[1] == 0:
                return False
        return True
    except Exception:
        return False

def _scan(cls_dir: Path):
    valid = []
    for p in cls_dir.iterdir():
        if is_valid_image(p):
            valid.append(p)
        else:
            p.unlink()
    return valid

def _copy(cls_dir: Path, paths, out_dir: Path, split: str):
    dst = out_dir / split / cls_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, dst / p.name)

def split_dataset(raw_dir=config.RAW_DIR, out_dir=config.DATA_DIR,
                  val_ratio=config.VAL_RATIO, seed=config.SEED):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    random.seed(seed)
    train_src, test_src = raw_dir / "train", raw_dir / "test"
    classes = sorted(d.name for d in train_src.iterdir() if d.is_dir())
    for cls in classes:
        train_cls, test_cls = train_src / cls, test_src / cls
        valid = _scan(train_cls)
        n_val = int(len(valid) * val_ratio)
        random.shuffle(valid)
        _copy(train_cls, valid[:n_val], out_dir, "val")
        _copy(train_cls, valid[n_val:], out_dir, "train")
        _copy(test_cls, _scan(test_cls), out_dir, "test")
    return classes

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def get_transform(augment: bool):
    from torchvision import transforms
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

def get_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE):
    from torch.utils.data import DataLoader
    from torchvision import datasets
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train",
                                    transform=get_transform(augment=True))
    val_ds = datasets.ImageFolder(data_dir / "val",
                                  transform=get_transform(augment=False))
    test_ds = datasets.ImageFolder(data_dir / "test",
                                   transform=get_transform(augment=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS)
    classes = train_ds.classes 
    num_classes = len(classes)
    return train_loader, val_loader, test_loader, classes, num_classes

if __name__ == "__main__":
    split_dataset()