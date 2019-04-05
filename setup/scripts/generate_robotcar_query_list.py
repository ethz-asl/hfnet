from pathlib import Path
from tqdm import tqdm


if __name__ == '__main__':
    images_dir = 'images'
    query_list_name = 'queries/{}_queries_with_intrinsics.txt'
    intrinsics_name = 'intrinsics/{}_intrinsics.txt'
    sequence = 'night-rain'
    h, w = 1024, 1024

    intrinsics = {}
    for side in ['left', 'right', 'rear']:
        with open(intrinsics_name.format(side), 'r') as f:
            fx = f.readline().split()[1]
            fy = f.readline().split()[1]
            cx = f.readline().split()[1]
            cy = f.readline().split()[1]
            assert fx == fy
            params = ['SIMPLE_RADIAL', w, h, fx, cx, cy, 0.0]
            intrinsics[side] = [str(p) for p in params]

    query_file = open(query_list_name.format(sequence), 'w')
    paths = sorted([p for p in Path(images_dir, sequence).glob('**/*.jpg')])
    for p in tqdm(paths):
        name = str(Path(p).relative_to(images_dir))
        side = Path(p).parent.name
        query_file.write(' '.join([name]+intrinsics[side])+'\n')
    query_file.close()
