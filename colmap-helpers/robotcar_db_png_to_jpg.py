import argparse
import numpy as np
import os
import sqlite3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_file", required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    connection = sqlite3.connect(args.db_file)
    cursor = connection.cursor()

    cursor.execute("SELECT image_id, name FROM images;")
    image_ids = []
    for row in cursor:
        image_ids.append((int(row[0]), row[1]))

    cursor.close()

    print 'Got', len(image_ids), 'image ids.'

    cursor = connection.cursor()
    for image_id_and_name in image_ids:
        jpg_name = os.path.splitext(image_id_and_name[1])[0] + '.jpg'
        new_params = [jpg_name, image_id_and_name[0]]
        cursor.execute('UPDATE images SET name = ? WHERE image_id = ?;', new_params)

    cursor.close()

    connection.commit()
    connection.close()


if __name__ == "__main__":
    main()
