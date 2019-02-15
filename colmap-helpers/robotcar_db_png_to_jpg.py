import argparse
import os
import sqlite3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_file', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    connection = sqlite3.connect(args.db_file)
    cursor = connection.cursor()

    cursor.execute('SELECT image_id, name FROM images;')
    ids_and_names = [(int(image_id), name) for image_id, name in cursor]
    print('Got', len(ids_and_names), 'image ids.')

    for image_id, name in ids_and_names:
        jpg_name = os.path.splitext(name)[0] + '.jpg'
        cursor.execute('UPDATE images SET name = ? WHERE image_id = ?;',
                       [jpg_name, image_id])

    cursor.close()
    connection.commit()
    connection.close()


if __name__ == '__main__':
    main()
