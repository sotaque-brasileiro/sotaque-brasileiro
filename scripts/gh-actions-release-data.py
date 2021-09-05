import os
import datetime
import zipfile

import sotaque_brasileiro


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, _, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


def get_today_date():
    return datetime.date.today().strftime("%Y%m%d")


sotaque_brasileiro.download_dataset(show_progress=False)

zipf = zipfile.ZipFile(
    f'sotaque-brasileiro-{get_today_date()}.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir(sotaque_brasileiro.constants.DATASET_SAVE_DEFAULT_PATH.value, zipf)
zipf.close()
