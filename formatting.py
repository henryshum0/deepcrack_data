from pathlib import Path

def format_path(img_dir=None, label_dir=None, label_postfix='', out_prefix='', out_postfix='', 
                out_img_postfix='', out_mask_postfix=''):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory {img_dir} does not exist.")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory {label_dir} does not exist.")
    
    img_files = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    label_files = sorted([f for f in label_dir.glob("*") if f.suffix.lower() in ['.png']])
    
    for img_file in img_files:
        img_name = img_file.stem
        img_file.rename(img_dir / f"{out_prefix}{img_name}{out_postfix}{out_img_postfix}.jpg")
    for label_file in label_files:
        label_name = label_file.stem
        if label_name.endswith(label_postfix):
            label_name = label_name[:-len(label_postfix)]
        label_file.rename(label_dir / f"{out_prefix}{label_name}{out_postfix}{out_mask_postfix}.png")

if __name__ == "__main__":
    img_dir = "data/images_ts"
    label_dir = "data/labels_ts"
    label_postfix = "_GT"
    out_prefix = "Crack_"
    out_img_postfix = "_0000"
    out_mask_postfix = ""

    format_path(img_dir, label_dir, label_postfix, out_prefix, '', out_img_postfix, out_mask_postfix)        