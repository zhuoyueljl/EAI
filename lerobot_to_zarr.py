import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc


ARRAY_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
IMAGE_COMPRESSOR = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_video(video_path: Path, expected_frames: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    data = np.stack(frames, axis=0).astype(np.uint8, copy=False)
    if expected_frames is not None and data.shape[0] != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch for {video_path}: decoded {data.shape[0]}, expected {expected_frames}"
        )
    return data


def convert_episode(parquet_path: Path, episode_group: zarr.Group, source_root: Path):
    table = pq.read_table(parquet_path)
    row_count = table.num_rows

    obs = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    act = np.asarray(table["action"].to_pylist(), dtype=np.float32)

    episode_group.create_array(
        "observation.state",
        data=obs,
        chunks=(min(256, row_count), obs.shape[1]),
        compressor=ARRAY_COMPRESSOR,
    )
    episode_group.create_array(
        "action",
        data=act,
        chunks=(min(256, row_count), act.shape[1]),
        compressor=ARRAY_COMPRESSOR,
    )

    for name, dtype in [
        ("timestamp", np.float32),
        ("frame_index", np.int64),
        ("frame_id", np.int64),
        ("episode_index", np.int64),
        ("index", np.int64),
        ("task_index", np.int64),
    ]:
        values = np.asarray(table[name].to_pylist(), dtype=dtype)
        episode_group.create_array(
            name,
            data=values,
            chunks=(min(512, row_count),),
            compressor=ARRAY_COMPRESSOR,
        )

    task_text = np.asarray(table["human.task_description"].to_pylist(), dtype=np.str_)
    episode_group.create_array(
        "human.task_description",
        data=task_text,
        chunks=(min(512, row_count),),
    )

    relative = parquet_path.relative_to(source_root)
    episode_group.attrs["source_parquet"] = str(relative).replace("\\", "/")
    episode_group.attrs["num_frames"] = int(row_count)

    episode_index = int(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)[0])
    chunk_id = episode_index // 1000
    videos_group = episode_group.create_group("videos")

    for video_key in ["observation.images.front", "observation.images.wrist"]:
        video_path = (
            source_root
            / "videos"
            / f"chunk-{chunk_id:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )
        frames = decode_video(video_path, expected_frames=row_count)
        dataset = videos_group.create_array(
            video_key,
            data=frames,
            chunks=(1, frames.shape[1], frames.shape[2], frames.shape[3]),
            compressor=IMAGE_COMPRESSOR,
        )
        dataset.attrs["source_video"] = str(video_path.relative_to(source_root)).replace("\\", "/")
        dataset.attrs["color_format"] = "RGB"
        dataset.attrs["shape_actual"] = list(map(int, frames.shape[1:]))


def main():
    parser = argparse.ArgumentParser(description="Convert a LeRobot dataset folder to Zarr.")
    parser.add_argument("--source", required=True, help="ASCII-safe source dataset directory")
    parser.add_argument("--output", required=True, help="Output .zarr directory")
    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)

    if output_root.exists():
        shutil.rmtree(output_root)

    info = read_json(source_root / "meta" / "info.json")
    modality = read_json(source_root / "meta" / "modality.json")
    stats = read_json(source_root / "meta" / "stats.json")
    episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    tasks = read_jsonl(source_root / "meta" / "tasks.jsonl")

    root = zarr.open_group(str(output_root), mode="w", zarr_format=2)
    root.attrs.update(
        {
            "source_root": str(source_root),
            "dataset_name": source_root.name,
            "format_note": "Converted from LeRobot dataset layout to episode-wise Zarr groups",
            "info": info,
            "modality": modality,
            "stats": stats,
            "episodes_meta": episodes,
            "tasks_meta": tasks,
        }
    )

    meta_group = root.create_group("meta")
    meta_group.create_array(
        "episode_lengths",
        data=np.asarray([ep["length"] for ep in episodes], dtype=np.int64),
        chunks=(min(256, len(episodes)),),
        compressor=ARRAY_COMPRESSOR,
    )
    meta_group.create_array(
        "episode_indices",
        data=np.asarray([ep["episode_index"] for ep in episodes], dtype=np.int64),
        chunks=(min(256, len(episodes)),),
        compressor=ARRAY_COMPRESSOR,
    )

    episode_root = root.create_group("episodes")
    parquet_paths = sorted((source_root / "data").glob("chunk-*/*.parquet"))
    for idx, parquet_path in enumerate(parquet_paths, start=1):
        episode_name = parquet_path.stem
        print(f"[{idx}/{len(parquet_paths)}] converting {episode_name}")
        episode_group = episode_root.create_group(episode_name)
        convert_episode(parquet_path, episode_group, source_root)

    print(f"Conversion complete: {output_root}")


if __name__ == "__main__":
    main()
