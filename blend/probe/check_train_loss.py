import sys, json
sys.path.insert(0, "blend/probe")
from nets import *
torch.set_num_threads(2)
R = "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend/probe"
pairs = json.load(open(R + "/pairs.json"))
tr = [k for k, p in enumerate(pairs) if p["split"] == "train"]
P = {k: Pair(R, k) for k in tr}
rng = np.random.default_rng(123)
plan = [(tr[rng.integers(56)], int(rng.integers(0, H - 128)), int(rng.integers(0, W - 128))) for _ in range(160)]
items = [P[k].window(y - 2, y + 130, x - 2, x + 130) for k, y, x in plan]
nets = {"init": Net("px8")}
for n in sys.argv[1:]:
    nets[n] = Net(n)
    nets[n].load_state_dict(torch.load(R + f"/nets/{n}.pt"))
with torch.no_grad():
    for n, net in nets.items():
        tot = 0
        mse = 0
        for i in range(0, 160, 16):
            X, Rr, Y = [torch.from_numpy(np.stack([it[j] for it in items[i:i + 16]])) for j in range(3)]
            c = mix(net(X, Rr), X)[:, :, 2:-2, 2:-2]
            y = Y[:, :, 2:-2, 2:-2]
            tot += torch.sqrt((c - y) ** 2 + 1e-6).mean().item()
            mse += ((c - y) ** 2).mean().item()
        print(n, "charb %.5f" % (tot / 10), "psnr %.2f" % (10 * np.log10(1 / (mse / 10))))
