import multiprocessing
import csv
import tqdm
import argparse
from pathlib import Path
from urllib import request
from PIL import Image
from io import BytesIO

from hfnet.settings import DATA_PATH


def parse_data(data_file, num=None):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    key_url_list = key_url_list[1:]  # Chop off header
    if num is not None:
        key_url_list = key_url_list[:num]
    return key_url_list


def download_image(key_url, output_dir):
    (key, url) = key_url
    filename = Path(output_dir, '{}.jpg'.format(key))

    if filename.exists():
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_rgb.save(str(filename), format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file', type=str, help='csv index file')
    parser.add_argument('--output_dir', action='store', type=str,
                        default=str(Path(DATA_PATH, 'google_landmarks/images')),
                        help='output directory')
    parser.add_argument('--truncate', action='store', default=None, type=int)
    parser.add_argument('--num_cpus', action='store', default=5, type=int)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    key_url_list = parse_data(args.index_file, args.truncate)

    def download_func(key_url):
        return download_image(key_url, args.output_dir)

    pool = multiprocessing.Pool(processes=args.num_cpus)
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_func, key_url_list),
                             total=len(key_url_list)))
    print('Total number of download failures: ', failures)
    pool.close()
    pool.terminate()
