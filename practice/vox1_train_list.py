from glob import iglob
import tqdm

root = '/mnt/E/sea120424/VoxCeleb_trainer/voxceleb1/*/*/*.wav'

f = sorted(iglob(root))
for path in f:
    ele = path.split('/')
    idx = ele[-3]
    uri = '/'.join(ele[-3:])
    print(idx, uri)
