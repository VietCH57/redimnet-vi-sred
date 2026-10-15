# Tiny fake Kaggle-like tree (synthetic "speakers", MUSAN, RIR) for smoke tests: python tests/make_fake_data.py <root>
import numpy as np, soundfile as sf, os, sys, random
R = sys.argv[1]; rng = np.random.default_rng(0); sr = 16000
K = f'{R}/kaggle/input/datasets'
tr = f'{K}/hongvitchugo/vi-sred26-train'; data = f'{tr}/Vi-SRED26-Train/data'
pt = f'{K}/hongvitchugo/vi-sred26-publictest'
def voice(spk, dur):
    f0 = 90 + 23 * spk; t = np.arange(int(dur * sr)) / sr
    y = sum(np.sin(2*np.pi*f0*h*t + rng.random()*6) / h for h in range(1, 12))
    y *= 0.5 + 0.5*np.sin(2*np.pi*(1.5 + 0.2*(spk % 5))*t)**2
    return (0.2 * y / np.abs(y).max() + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
def write(path, y): os.makedirs(os.path.dirname(path), exist_ok=True); sf.write(path, y, sr, subtype='PCM_16')
train, val = [], []
for s in range(14):
    sid = f'id{s*3:05d}'
    for u in range(1, 7):
        rel = f'{sid}/{u:04d}.wav'; write(f'{data}/{rel}', voice(s, rng.uniform(1.0, 5.0)))
        (train if s < 10 else val).append((sid, rel))
open(f'{tr}/train_split.txt','w').write(''.join(f'{a}\t{b}\n' for a,b in train))
open(f'{tr}/val_split.txt','w').write(''.join(f'{a}\t{b}\n' for a,b in val))
tri = []
for i in range(len(val)):
    for j in range(i+1, len(val)): tri.append((int(val[i][0]==val[j][0]), val[i][1], val[j][1]))
random.seed(0); random.shuffle(tri)
open(f'{tr}/val_trials.txt','w').write(''.join(f'{l} {a} {b}\n' for l,a,b in tri))
# public test: different speakers, label word last
pub = []
for s in range(20, 24):
    for u in range(1, 4):
        rel = f'id{s:05d}/{u:04d}.wav'; write(f'{pt}/data/{rel}', voice(s, rng.uniform(1.0, 4.0))); pub.append(rel)
lines = [f'{a} {b} {"target" if a[:7]==b[:7] else "nontarget"}' for i,a in enumerate(pub) for b in pub[i+1:]]
random.shuffle(lines); open(f'{pt}/public-test.txt','w').write('\n'.join(lines)+'\n')
# MUSAN (raw layout, long files) and RIRS_NOISES (double nested)
mu = f'{K}/nhattruongdev/musan-noise/musan'
for cat, sub in [('music','fma'),('noise','free-sound'),('speech','librivox')]:
    for k in range(9 if cat=='speech' else 3):
        write(f'{mu}/{cat}/{sub}/{cat}-{sub}-{k:04d}.wav', (0.1*rng.standard_normal(int(sr*rng.uniform(1, 20)))).astype(np.float32))
open(f'{mu}/README','w').write('fake')
rr = f'{K}/nhattruongdev/rirs-noises/RIRS_NOISES/RIRS_NOISES'
for room in ['smallroom/Room001','mediumroom/Room002']:
    for k in range(3):
        h = rng.standard_normal(4000) * np.exp(-np.arange(4000)/600); write(f'{rr}/simulated_rirs/{room}/{room[-7:]}-{k:05d}.wav', (0.5*h/np.abs(h).max()).astype(np.float32))
write(f'{rr}/pointsource_noises/noise-free-sound-0001.wav', rng.standard_normal(16000).astype(np.float32)*0.1)
write(f'{rr}/real_rirs_isotropic_noises/air_type1.wav', rng.standard_normal(8000).astype(np.float32)*0.1)
print('train', len(train), 'val', len(val), 'val trials', len(tri), 'public trials', len(lines))
