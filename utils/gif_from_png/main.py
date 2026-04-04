# create_gif_with_repeats.py

import os
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
from PIL import Image

def repeat_frames(frames: List[Image.Image], repeat: int) -> List[Image.Image]:
    """
    Given a list of PIL Images, return a new list where each frame
    is repeated `repeat` times (shallow-copied so palette info is preserved).
    """
    out = []
    for im in frames:
        for _ in range(repeat):
            out.append(im.copy())
    return out

@hydra.main(config_path="all/main", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # 1. Resolve input and output paths
    input_paths: List[Path] = [Path(f).expanduser() for f in cfg.files]
    output_path: Path = Path(cfg.output).expanduser()
    duration_ms: int    = int(cfg.duration)
    repeat_count: int   = int(cfg.repeat)  # how many times to repeat each frame

    # 2. Load PNGs into memory
    images = []
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        # convert to RGBA so transparency is kept if present
        images.append(Image.open(p).convert("RGBA"))

    # 3. Repeat frames to implement pauses
    frames = repeat_frames(images, repeat_count)

    # 4. Make sure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Save as animated GIF
    #    - save_all: include all frames
    #    - append_images: everything after the first
    #    - duration: how long (ms) each frame is shown
    #    - loop=0: infinite loop
    #    - disposal=2: clear previous frame before drawing next (avoids bleed-through)  [oai_citation_attribution:0‡Nkmk Note](https://note.nkmk.me/en/python-pillow-gif/?utm_source=chatgpt.com)
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False
    )

    print(f"Animated GIF written to {output_path}")

if __name__ == "__main__":
    main()