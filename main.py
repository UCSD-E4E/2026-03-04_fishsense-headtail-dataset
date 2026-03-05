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

from multiprocessing import get_context


_WORKER_CONTEXT = {}


def _init_worker(worker_context):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = worker_context


def _process_label_image(pair):
    label, image = pair

    dives_by_id = _WORKER_CONTEXT["dives_by_id"]
    cameras_by_id = _WORKER_CONTEXT["cameras_by_id"]
    camera_intrinsics_by_camera_id = _WORKER_CONTEXT["camera_intrinsics_by_camera_id"]
    data_folder = Path(_WORKER_CONTEXT["data_folder"])
    image_output_folder = Path(_WORKER_CONTEXT["image_output_folder"])
    label_output_folder = Path(_WORKER_CONTEXT["label_output_folder"])

    parsed_url = urlparse(_WORKER_CONTEXT["nas_url"])
    filestation = FileStation(
        parsed_url.hostname,
        parsed_url.port,
        _WORKER_CONTEXT["nas_username"],
        _WORKER_CONTEXT["nas_password"],
        secure=True,
        cert_verify=False,
    )

    dive = dives_by_id[image.dive_id]
    camera = cameras_by_id[dive.camera_id]
    camera_intrinsics = camera_intrinsics_by_camera_id[camera.id]

    image_path = data_folder / image.path
    image_target_path = image_output_folder / f"{image.checksum}.JPG"
    label_target_path = label_output_folder / f"{image.checksum}.txt"

    source_nas_path = f"/fishsense_data/REEF/data/{image.path}"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    filestation.get_file(source_nas_path, "download", dest_path=str(image_path.parent))

    rectified_image = RectifiedImage(RawImage(image_path), camera_intrinsics)
    cv2.imwrite(image_target_path.as_posix(), rectified_image.data)

    if label.head_x is None or label.head_y is None or label.tail_x is None or label.tail_y is None:
        return

    with open(label_target_path, "w") as f:
        f.write(f"{label.head_x} {label.head_y} {label.tail_x} {label.tail_y}\n")

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

    # %%
    worker_context = {
        "dives_by_id": dives_by_id,
        "cameras_by_id": cameras_by_id,
        "camera_intrinsics_by_camera_id": camera_intrinsics_by_camera_id,
        "data_folder": str(DATA_FOLDER),
        "image_output_folder": str(IMAGE_OUTPUT_FOLDER),
        "label_output_folder": str(LABEL_OUTPUT_FOLDER),
        "nas_url": settings.e4e_nas.url,
        "nas_username": settings.e4e_nas.username,
        "nas_password": settings.e4e_nas.password,
    }

    tasks = list(zip(labels, images))
    with get_context("spawn").Pool(processes=16, initializer=_init_worker, initargs=(worker_context,)) as pool:
        list(tqdm(pool.imap_unordered(_process_label_image, tasks), total=len(tasks)))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())