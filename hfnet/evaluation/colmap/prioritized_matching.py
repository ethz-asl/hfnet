import argparse
import os

import export_features
import db_matches
import match_frames

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--min_num_matches", type=int, default=20)
    # This argument lets us only look at the matches from a certain folder.
    parser.add_argument("--filter_image_dir", default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    matching_image_pairs = \
      db_matches.get_matching_images(args.database_path, args.min_num_matches,
                                     args.filter_image_dir)
    print 'Got ', len(matching_image_pairs), ' matching image pairs.'

    outfile = open("matches.txt", "w+")

    matched = 0
    for matching_pair in matching_image_pairs:
        # Get npz instead of image files.
        frame1 = os.path.splitext(matching_pair[0].encode('utf-8'))[0]+'.npz'
        frame2 = os.path.splitext(matching_pair[1].encode('utf-8'))[0]+'.npz'

        num_points = 1500
        keypoint_matches = match_frames.match_frames(frame1, frame2,
                                  'images_upright/' + matching_pair[0],
                                  'images_upright/' + matching_pair[1],
                                  num_points, debug = False)

        outfile.write(os.path.basename(matching_pair[0]) + ' ' + os.path.basename(matching_pair[1]) + '\n')
        for match in keypoint_matches:
            outfile.write(str(match[0]) + ' ' + str(match[1]) + '\n')

        outfile.write('\n')

        matched = matched + 1
        print 'Matched ', frame1, frame2, ':', format(float(matched) / len(matching_image_pairs) * 100, '.2f') + '% done.'

    outfile.close()


if __name__ == "__main__":
    main()
