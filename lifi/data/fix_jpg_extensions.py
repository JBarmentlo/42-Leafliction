from pathlib import Path


def fix_jpg_extensions(data_folder: str):
    folder = Path(data_folder)
    image_files = list(Path.cwd().glob(str(folder / "**/*.JPG")))
    i = 0
    for im in image_files:
        if im.suffix == ".JPG":
            im.rename(im.with_suffix(".jpg"))
            i += 1

    print(f"Fixed {i} image extenstion in {data_folder}.")
