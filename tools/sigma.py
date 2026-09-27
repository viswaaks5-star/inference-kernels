import re, statistics as st, sys
data, cfg = {}, None
for line in open(sys.argv[1]):
    m = re.match(r"^(\S.*?)\s+\d+\.\d{4}\s+\d+\.\d\s+\d+\.\d%", line)
    if m:
        cfg = m.group(1).strip(); data[cfg] = []; continue
    m = re.match(r"^\s+rep (\d+): (\d+\.\d+) ms", line)
    if m and cfg:
        data[cfg].append((int(m.group(1)), float(m.group(2))))
for cfg, reps in data.items():
    ms = [t for _, t in reps]
    med = st.median(ms)
    sigma = 1.4826 * st.median(abs(t - med) for t in ms) / med * 100
    reps.sort()
    half = len(reps) // 2
    early = st.median(t for _, t in reps[:half])
    late = st.median(t for _, t in reps[half:])
    drift = (late - early) / med * 100
    print(f"{cfg:32s} n={len(ms):2d}  sigma={sigma:.3f}%  3sigma={3*sigma:.3f}%  drift={drift:+.3f}%")
