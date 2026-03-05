from fishsense_api_sdk.client import Client
from fishsense_headtail_dataset.config import settings
from tqdm.asyncio import tqdm as tqdm_asyncio
from httpx import HTTPStatusError
from synology_api.filestation import FileStation
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import cv2
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage
import os

from multiprocessing import get_context


_WORKER_CONTEXT = {}
_FILESTATION = None


def _init_worker(worker_context):
    global _WORKER_CONTEXT
    global _FILESTATION
    _WORKER_CONTEXT = worker_context

    # Avoid each worker over-subscribing CPU via OpenMP/BLAS thread pools.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)

    parsed_url = urlparse(_WORKER_CONTEXT["nas_url"])
    _FILESTATION = FileStation(
        parsed_url.hostname,
        parsed_url.port,
        _WORKER_CONTEXT["nas_username"],
        _WORKER_CONTEXT["nas_password"],
        secure=True,
        cert_verify=False,
    )


def _process_label_image(task):
    camera_intrinsics_by_camera_id = _WORKER_CONTEXT["camera_intrinsics_by_camera_id"]
    data_folder = Path(_WORKER_CONTEXT["data_folder"])
    image_output_folder = Path(_WORKER_CONTEXT["image_output_folder"])
    label_output_folder = Path(_WORKER_CONTEXT["label_output_folder"])

    camera_intrinsics = camera_intrinsics_by_camera_id[task["camera_id"]]
    image_path = data_folder / task["relative_image_path"]
    image_target_path = image_output_folder / f"{task['checksum']}.JPG"
    label_target_path = label_output_folder / f"{task['checksum']}.txt"

    if image_target_path.exists():
        return os.getpid()

    source_nas_path = f"/fishsense_data/REEF/data/{task['relative_image_path']}"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    
    for _ in range(3):  # Retry up to 3 times
        try:
            _FILESTATION.get_file(source_nas_path, "download", dest_path=str(image_path.parent))
            break  # Success, exit the retry loop
        except Exception as e:
            print(f"Error downloading {source_nas_path}: {e}")

        return os.getpid()

    rectified_image = RectifiedImage(RawImage(image_path), camera_intrinsics)
    cv2.imwrite(image_target_path.as_posix(), rectified_image.data)

    if task["head_x"] is None or task["head_y"] is None or task["tail_x"] is None or task["tail_y"] is None:
        return os.getpid()

    with open(label_target_path, "w") as f:
        f.write(f"{task['head_x']} {task['head_y']} {task['tail_x']} {task['tail_y']}\n")

    return os.getpid()

async def main():
    # %%
    DATA_FOLDER = (Path("./data") / "REEF" / "data").absolute()
    OUTPUT_FOLDER = (Path("./output") / "headtail_dataset").absolute()

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    IMAGE_OUTPUT_FOLDER = OUTPUT_FOLDER / "images"
    LABEL_OUTPUT_FOLDER = OUTPUT_FOLDER / "labels"

    IMAGE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    LABEL_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    DATA_FOLDER.exists(), OUTPUT_FOLDER.exists()

    # %%
    async with Client(settings.fishsense_api.url, settings.fishsense_api.username, settings.fishsense_api.password) as client:
        dives = await client.dives.get_canonical()

    len(dives), dives

    # %%
    dives_by_id = {dive.id: dive for dive in dives}

    len(dives_by_id), dives_by_id

    # %%
    async def get_headtail_labels(client, dive):
        try:
            return await client.labels.get_headtail_labels(dive.id)
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    # %%
    async with Client(settings.fishsense_api.url, settings.fishsense_api.username, settings.fishsense_api.password) as client:
        label_lists = await tqdm_asyncio.gather(*[get_headtail_labels(client, dive) for dive in dives])

    label_lists = [label_list for label_list in label_lists if label_list is not None]
    labels = [label for label_list in label_lists for label in label_list if label is not None]
    len(labels), labels

    # %%
    async with Client(settings.fishsense_api.url, settings.fishsense_api.username, settings.fishsense_api.password) as client:
        images = await tqdm_asyncio.gather(*[client.images.get(image_id=label.image_id) for label in labels])

    len(images), images

    # %%
    async with Client(settings.fishsense_api.url, settings.fishsense_api.username, settings.fishsense_api.password) as client:
        cameras = await tqdm_asyncio.gather(*[client.cameras.get(dive.camera_id) for dive in dives])

    len(cameras), cameras

    # %%
    cameras_by_id = {camera.id: camera for camera in cameras}

    len(cameras_by_id), cameras_by_id

    # %%
    async with Client(settings.fishsense_api.url, settings.fishsense_api.username, settings.fishsense_api.password) as client:
        camera_intrinsics_list = await tqdm_asyncio.gather(*[client.cameras.get_intrinsics(camera.id) for camera in cameras])

    len(camera_intrinsics_list), camera_intrinsics_list

    # %%
    camera_intrinsics_by_camera_id = {intrinsics.camera_id: intrinsics for intrinsics in camera_intrinsics_list}

    len(camera_intrinsics_by_camera_id), camera_intrinsics_by_camera_id

    dive_camera_id_by_dive_id = {dive.id: dive.camera_id for dive in dives}

    # %%
    worker_context = {
        "camera_intrinsics_by_camera_id": camera_intrinsics_by_camera_id,
        "data_folder": str(DATA_FOLDER),
        "image_output_folder": str(IMAGE_OUTPUT_FOLDER),
        "label_output_folder": str(LABEL_OUTPUT_FOLDER),
        "nas_url": settings.e4e_nas.url,
        "nas_username": settings.e4e_nas.username,
        "nas_password": settings.e4e_nas.password,
    }

    tasks = []
    for label, image in zip(labels, images):
        tasks.append(
            {
                "camera_id": dive_camera_id_by_dive_id[image.dive_id],
                "relative_image_path": image.path,
                "checksum": image.checksum,
                "head_x": label.head_x,
                "head_y": label.head_y,
                "tail_x": label.tail_x,
                "tail_y": label.tail_y,
            }
        )

    with get_context("spawn").Pool(processes=16, initializer=_init_worker, initargs=(worker_context,)) as pool:
        worker_pids = list(tqdm(pool.imap_unordered(_process_label_image, tasks, chunksize=1), total=len(tasks)))

    unique_workers = len(set(worker_pids))
    print(f"Processed {len(worker_pids)} tasks using {unique_workers} worker processes.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())