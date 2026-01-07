from pathlib import Path
from PIL import Image

def to_white_bg(in_path: str | Path, out_path: str | Path) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)

    im = Image.open(in_path)
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
        im = im.convert("RGBA")
        white = Image.new("RGBA", im.size, (255, 255, 255, 255))
        out = Image.alpha_composite(white, im).convert("RGB")
    else:
        out = im.convert("RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
