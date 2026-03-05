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
    parsed_url = urlparse(settings.e4e_nas.url)

    filestation = FileStation(parsed_url.hostname, parsed_url.port, settings.e4e_nas.username, settings.e4e_nas.password, secure=True, cert_verify=False)

    # %%
    for label, image in tqdm(zip(labels, images), total=len(labels)):
        dive = dives_by_id[image.dive_id]
        camera = cameras_by_id[dive.camera_id]
        camera_intrinsics = camera_intrinsics_by_camera_id[camera.id]

        image_path = DATA_FOLDER / image.path
        image_target_path = IMAGE_OUTPUT_FOLDER / f"{image.checksum}.JPG"
        label_target_path = LABEL_OUTPUT_FOLDER / f"{image.checksum}.txt"

        source_nas_path = f"/fishsense_data/REEF/data/{image.path}"
        filestation.get_file(source_nas_path, "download", dest_path=str(image_path.parent))

        image = RectifiedImage(RawImage(image_path), camera_intrinsics)
        img = image.data

        cv2.imwrite(image_target_path.as_posix(), img)

        if label.head_x is None or label.head_y is None or label.tail_x is None or label.tail_y is None:
            continue
        
        with open(label_target_path, "w") as f:
            f.write(f"{label.head_x} {label.head_y} {label.tail_x} {label.tail_y}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())