import re, sys
names = {1:"non-IDR", 5:"IDR", 6:"SEI", 7:"SPS", 8:"PPS", 9:"AUD"}
d = open(sys.argv[1], "rb").read()
sc = [m.start() for m in re.finditer(b"\x00\x00\x00\x01", d)]
print("size=%d bytes, 起始码=%d 个" % (len(d), len(sc)))
counts = {}
for i, p in enumerate(sc):
    t = d[p+4] & 0x1f
    end = sc[i+1] if i+1 < len(sc) else len(d)
    counts[t] = counts.get(t, 0) + 1
    print("  off=%6d type=%2d %-8s len=%d" % (p, t, names.get(t, "?"), end-p))
print("汇总:", {names.get(k,k): v for k, v in sorted(counts.items())})
