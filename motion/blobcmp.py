import numpy as np, glob, sys
# blobcmp.py DIR_TODAY TAG_TODAY DIR_NEW TAG_NEW : per pair blob counts (3x3 median, > 4 flow px, 8 px blocks), non-driving pairs
def med(a):
    p = np.pad(a, 1, mode="edge"); h, w = a.shape
    return np.median(np.stack([p[i:i + h, j:j + w] for i in range(3) for j in range(3)]), 0)
def count(f):
    d = np.load(f); F = d['F'].astype(np.float32); fs = int(d['fs'])
    H, W = F.shape[:2]; b = 8
    B = F[:H // b * b, :W // b * b].reshape(H // b, b, W // b, b, 2).mean((1, 3)) * fs
    M = np.stack([med(B[..., i]) for i in range(2)], -1)
    return int((np.abs(B - M).max(-1) > 4).sum())
def get(d, t):
    r = {}
    for f in glob.glob(d + '/fields/*_' + t + '.npz'):
        r[f.split('/')[-1].rsplit('_', 1)[0]] = f
    return r
A, B = get(sys.argv[1], sys.argv[2]), get(sys.argv[3], sys.argv[4])
DRV = ('racing', 'road_', 'forza', 'faroe')
st = sa = sb = nbad = 0
for k in sorted(A):
    if k not in B or k.startswith(DRV): continue
    a, b = count(A[k]), count(B[k]); sa += a; sb += b
    bad = b > a + 2; nbad += bad
    print("%-44s today %5d new %5d %s" % (k, a, b, "WORSE" if bad else ""))
print("non-driving sum today %d new %d, pairs above today+2: %d" % (sa, sb, nbad))
