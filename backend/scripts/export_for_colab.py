"""Export a dataset in YOLO-seg format, zipped and ready to upload to the
train_on_colab.ipynb notebook (colab/) for training on a free Colab GPU.

Usage:
    cd backend
    python scripts/export_for_colab.py <dataset_name> [output_zip_path]
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import dataset_service  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_name = sys.argv[1]
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / f"{dataset_name}-dataset.zip"

    data_yaml = dataset_service.export_yolo_dataset(dataset_name)
    export_dir = data_yaml.parent

    # the exported data.yaml has an absolute local path baked in; rewrite it
    # to "." so the zip is self-contained regardless of where it's extracted.
    text = data_yaml.read_text()
    lines = [("path: .\n" if line.startswith("path:") else line) for line in text.splitlines(keepends=True)]
    data_yaml.write_text("".join(lines))

    zip_base = str(out_path.with_suffix(""))
    shutil.make_archive(zip_base, "zip", export_dir)
    print(f"Exported '{dataset_name}' -> {zip_base}.zip")


if __name__ == "__main__":
    main()
