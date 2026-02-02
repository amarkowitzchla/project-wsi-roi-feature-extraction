from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys

from wsi_pipeline.utils.common import Manifest, ensure_dir, utc_now, write_json


class DeepZoomError(RuntimeError):
    pass


def _deepzoom_tile_script_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "example_notebook" / "deepzoom_tile.py"
    return script_path if script_path.exists() else None


def build_deepzoom(
    slide_id: str,
    slide_path: Path,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    limit_bounds: bool,
    manifest_path: Path,
    force: bool,
) -> None:
    ensure_dir(output_dir)
    dzi_path = output_dir / f"{slide_id}.dzi"
    tiles_dir = output_dir / "tiles"

    if dzi_path.exists() and tiles_dir.exists() and not force:
        return

    script_path = _deepzoom_tile_script_path()
    if script_path is not None:
        basename = output_dir / slide_id
        cmd = [
            sys.executable,
            str(script_path),
            "--format",
            "png",
            "--size",
            str(tile_size),
            "--overlap",
            str(overlap),
            "--output",
            str(basename),
            "--jobs",
            str(max(1, (os.cpu_count() or 2) - 1)),
            "--quality",
            "95",
        ]
        if not limit_bounds:
            cmd.append("--ignore-bounds")
        cmd.append(str(slide_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise DeepZoomError(
                "DeepZoom tiling failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        generated_tiles = output_dir / f"{slide_id}_files"
        if generated_tiles.exists():
            if tiles_dir.exists():
                shutil.rmtree(tiles_dir)
            generated_tiles.rename(tiles_dir)
        if not dzi_path.exists():
            src_dzi = output_dir / f"{slide_id}.dzi"
            if src_dzi.exists():
                shutil.move(str(src_dzi), str(dzi_path))
    else:
        try:
            import openslide
            from openslide.deepzoom import DeepZoomGenerator
        except Exception as exc:
            hint = (
                "OpenSlide could not be loaded. Install it before running deepzoom.\n"
                "Suggested micromamba install:\n"
                "  micromamba install -c conda-forge openslide openslide-python libxml2 icu\n"
            )
            raise DeepZoomError(f"{hint}\nOriginal error: {exc}") from exc

        slide = openslide.open_slide(str(slide_path))
        dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds)

        ensure_dir(tiles_dir)
        for level in range(dz.level_count):
            cols, rows = dz.level_tiles[level]
            level_dir = tiles_dir / str(level)
            ensure_dir(level_dir)
            for row in range(rows):
                for col in range(cols):
                    tile = dz.get_tile(level, (col, row))
                    tile_path = level_dir / f"{col}_{row}.png"
                    tile.save(tile_path, format="PNG")

        dzi_content = dz.get_dzi("png")
        dzi_path.write_text(dzi_content, encoding="utf-8")

    manifest = Manifest(
        name="deepzoom_manifest",
        created_at=utc_now(),
        version="1.0",
        params={
            "tile_size": tile_size,
            "overlap": overlap,
            "limit_bounds": limit_bounds,
        },
        inputs={"slide_id": slide_id, "slide_path": str(slide_path)},
        outputs={"dzi": str(dzi_path), "tiles_dir": str(tiles_dir)},
    )
    write_json(manifest_path, manifest.to_dict())
