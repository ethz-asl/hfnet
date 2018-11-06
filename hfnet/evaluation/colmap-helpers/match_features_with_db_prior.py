import argparse
import os

from internal import db_matching_images
from internal import frame_matching


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_file", required=True)
    parser.add_argument("--min_num_matches", type=int, default=15)
    parser.add_argument("--num_points_per_frame", type=int, default=2500)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--npz_dir", required=True)

    # This argument lets us only look at the matches from a certain folder.
    # We want to avoid adding matches from other folders, e.g. query. This
    # filters images according to the prefix as stored in the db file.
    parser.add_argument("--image_prefix", required=True)

    parser.add_argument("--output_file", required=True)
    parser.add_argument('--use_ratio_test', dest='use_ratio_test', default=True,
                        action='store_true')
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    matching_image_pairs = \
      db_matching_images.get_matching_images(args.database_file,
                                             args.min_num_matches,
                                             args.image_prefix)
    print 'Got', len(matching_image_pairs), 'matching image pairs.'

    outfile = open(args.output_file, "w+")

    matched = 0
    for matching_pair in matching_image_pairs:
        # Get npz instead of image files.
        npz1 = os.path.join(args.npz_dir, os.path.splitext(os.path.basename(matching_pair[0].encode('utf-8')))[0] + '.npz')
        npz2 = os.path.join(args.npz_dir, os.path.splitext(os.path.basename(matching_pair[1].encode('utf-8')))[0] + '.npz')

        image1 = os.path.join(args.image_dir, matching_pair[0])
        image2 = os.path.join(args.image_dir, matching_pair[1])

        assert(os.path.isfile(npz1))
        assert(os.path.isfile(npz2))
        assert(os.path.isfile(image1))
        assert(os.path.isfile(image2))

        num_points = args.num_points_per_frame
        keypoint_matches = frame_matching.match_frames(npz1, npz2, image1,
                                                       image2, num_points,
                                                       args.use_ratio_test,
                                                       args.debug)

        outfile.write(matching_pair[0] + ' ' + matching_pair[1] + '\n')
        for match in keypoint_matches:
            outfile.write(str(match[0]) + ' ' + str(match[1]) + '\n')

        outfile.write('\n')

        matched = matched + 1
        percentage = float(matched) / len(matching_image_pairs) * 100
        print 'Matched', npz1, 'with', npz2, ':', format(percentage, '.2f') \
              + '% done.'

    outfile.close()


if __name__ == "__main__":
    main()
