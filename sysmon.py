#!/usr/bin/env python3
"""
系统状态采集 —— 给网页那条实时状态栏喂数据。

只读 /proc、/sys、（可选）debugfs，不写任何东西。

采样跑在【一个后台线程】里，1 Hz 刷新一份快照，HTTP 那边只读快照：
  · CPU 利用率必须靠两次 /proc/stat 求差。多个客户端各采各的就会互相吃掉差值，
    一个页面开着、另一个页面刷新一下，CPU% 就跳成 0 —— 所以必须单点采样。
  · 顺带让每个 SSE 连接只做「读快照 + 序列化」，不重复打系统调用。

★读不到 ≠ 0。GPU / NPU 的硬件计数器在 debugfs 里，默认 600 root-only；
  本服务以普通用户跑，读不到时如实返回 None，前端显示「不可用」。
  绝不能把「读不到」写成 0% —— 那看起来和「硬件空闲」一模一样，是骗人。
"""
import os
import re
import threading
import time

# ---------------------------------------------------------------- 温度
# A733 一共 8 个 zone。带 _idle_ 的是给热管理当参考基准的区，
# 实测它们常年比对应的主传感器还高 1~3°C，所以「全场最高」经常命中 idle 区——
# 那是真读数、不是错，名字照实标出来就行。
_TEMP_LABEL = {
    "cpul_thermal_zone": ("CPU 小核", "主"),
    "cpub_thermal_zone": ("CPU 大核", "主"),
    "gpu_thermal_zone":  ("GPU", "主"),
    "npu_thermal_zone":  ("NPU", "主"),
    "ddr_thermal_zone":  ("DDR", "主"),
    "skin_zone":         ("板面", "主"),
    "cpul_idle_zone":    ("小核·空闲参考", "参考"),
    "cpub_idle_zone":    ("大核·空闲参考", "参考"),
}
_THERMAL = "/sys/class/thermal"

# ARM CPU part -> 核型号（只列常见的，认不出就退回十六进制）
_ARM_PARTS = {
    "0xd03": "Cortex-A53", "0xd05": "Cortex-A55", "0xd07": "Cortex-A57",
    "0xd08": "Cortex-A72", "0xd0b": "Cortex-A76", "0xd0d": "Cortex-A77",
    "0xd41": "Cortex-A78", "0xd44": "Cortex-X1",  "0xd46": "Cortex-A510",
    "0xd47": "Cortex-A710", "0xd48": "Cortex-X2", "0xd4d": "Cortex-A715",
    "0xd4e": "Cortex-X3",  "0xd81": "Cortex-A720", "0xd84": "Cortex-X4",
}

# GPU / NPU 计数器：debugfs 路径（读不到就 None）
_PVR_STATUS = "/sys/kernel/debug/pvr/status"
_VIP_LOAD   = "/sys/kernel/debug/viplite/core_loading"


def _read(path):
    """读一个小文本文件。任何失败（不存在/权限/竞态）都返回 None。"""
    try:
        with open(path, "rb") as f:
            return f.read(64 << 10).decode("utf-8", "replace")
    except OSError:
        return None


def _read_int(path):
    s = _read(path)
    if s is None:
        return None
    try:
        return int(s.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------- 温度
_trip_cache = {}          # zone 名 -> (最低阈值, 最高阈值)，开机后不变，只读一次


def _trips(zone_dir, typ):
    """(最低 trip, 最高 trip)，单位 °C。trip 点开机后不变，缓存。

    最低的那个 trip 就是热管理【开始动手】的温度（A733 上 CPU 是 60°C，
    板面 skin 只有 50°C）—— 前端拿它当进度条满量程，比绝对温度有意义得多。"""
    if typ in _trip_cache:
        return _trip_cache[typ]
    vals = []
    try:
        for f in os.listdir(zone_dir):
            if f.startswith("trip_point_") and f.endswith("_temp"):
                v = _read_int(os.path.join(zone_dir, f))
                if v:
                    vals.append(v / 1000.0)
    except OSError:
        pass
    r = (min(vals), max(vals)) if vals else (None, None)
    if vals:                                   # 只在读到时才缓存，别把失败固化
        _trip_cache[typ] = r
    return r


def temps():
    out = []
    try:
        names = sorted(os.listdir(_THERMAL))
    except OSError:
        return out
    for n in names:
        if not n.startswith("thermal_zone"):
            continue
        z = os.path.join(_THERMAL, n)
        t = _read_int(os.path.join(z, "temp"))
        if t is None:
            continue
        typ = (_read(os.path.join(z, "type")) or n).strip()
        label, kind = _TEMP_LABEL.get(typ, (typ, "其它"))
        lo, hi = _trips(z, typ)
        out.append({"zone": typ, "label": label, "kind": kind,
                    "c": round(t / 1000.0, 1),
                    "trip": lo, "crit": hi})
    out.sort(key=lambda x: -x["c"])
    return out


# ---------------------------------------------------------------- CPU
def _cpu_times():
    """{cpu号: [user,nice,sys,idle,iowait,irq,softirq,steal], "all": [...]}"""
    out = {}
    s = _read("/proc/stat")
    if not s:
        return out
    for ln in s.splitlines():
        if not ln.startswith("cpu"):
            break
        p = ln.split()
        key = p[0][3:]
        if key and not key.isdigit():
            continue
        try:
            v = [int(x) for x in p[1:9]]
        except ValueError:
            continue
        if len(v) < 8:
            v += [0] * (8 - len(v))
        out["all" if key == "" else int(key)] = v
    return out


def _cpu_model_map():
    """cpu号 -> 核型号。从 /proc/cpuinfo 的 CPU part 取。"""
    m, cur = {}, None
    s = _read("/proc/cpuinfo")
    if not s:
        return m
    for ln in s.splitlines():
        k, _, v = ln.partition(":")
        k, v = k.strip(), v.strip()
        if k == "processor":
            try:
                cur = int(v)
            except ValueError:
                cur = None
        elif k == "CPU part" and cur is not None:
            m[cur] = _ARM_PARTS.get(v.lower(), "ARM " + v)
    return m


def _cluster_of():
    """cpu号 -> cluster 号（按 cpufreq policy 分组：同 policy 即同簇）。"""
    rid, out = {}, {}
    base = "/sys/devices/system/cpu"
    try:
        cpus = [d for d in os.listdir(base)
                if d.startswith("cpu") and d[3:].isdigit()]
    except OSError:
        return out
    for c in sorted(cpus, key=lambda x: int(x[3:])):
        pol = os.path.realpath(os.path.join(base, c, "cpufreq"))
        if not os.path.isdir(pol):
            continue
        out[int(c[3:])] = rid.setdefault(pol, len(rid))
    return out


def cpu_stat(prev, cur):
    """把两次 _cpu_times() 求差，得到每核利用率。prev 为 None 时只回频率。"""
    util = {}
    if prev:
        for k in cur:
            if k not in prev:
                continue
            a, b = prev[k], cur[k]
            # idle = idle + iowait；其余算忙。
            da = sum(b) - sum(a)
            di = (b[3] + b[4]) - (a[3] + a[4])
            if da > 0:
                util[k] = max(0.0, min(100.0, (da - di) * 100.0 / da))
    return util


def cpus():
    base = "/sys/devices/system/cpu"
    model = _cpu_model_map()
    clus = _cluster_of()
    out = []
    for c in sorted(clus):
        d = os.path.join(base, "cpu%d" % c)
        f = _read_int(os.path.join(d, "cpufreq/scaling_cur_freq"))
        fmax = _read_int(os.path.join(d, "cpufreq/scaling_max_freq"))
        fmin = _read_int(os.path.join(d, "cpufreq/scaling_min_freq"))
        gov = (_read(os.path.join(d, "cpufreq/scaling_governor")) or "").strip()
        online = (_read(os.path.join(d, "online")) or "1").strip()
        out.append({
            "cpu": c, "model": model.get(c, ""), "cluster": clus[c],
            "mhz": None if f is None else round(f / 1000.0),
            "max_mhz": None if fmax is None else round(fmax / 1000.0),
            "min_mhz": None if fmin is None else round(fmin / 1000.0),
            "gov": gov, "online": online != "0",
        })
    return out


# ---------------------------------------------------------------- 内存
def mem():
    s = _read("/proc/meminfo")
    if not s:
        return None
    d = {}
    for ln in s.splitlines():
        k, _, v = ln.partition(":")
        v = v.strip().split()
        if v:
            try:
                d[k] = int(v[0])
            except ValueError:
                pass
    tot = d.get("MemTotal")
    avail = d.get("MemAvailable")
    if not tot:
        return None
    used = (tot - avail) if avail is not None else (tot - d.get("MemFree", 0))
    sw_t, sw_f = d.get("SwapTotal", 0), d.get("SwapFree", 0)
    kb = lambda x: round((x or 0) / 1024.0, 1)          # kB -> MB
    return {
        "total_mb": kb(tot), "used_mb": kb(used), "avail_mb": kb(avail),
        "cached_mb": kb(d.get("Cached")), "buffers_mb": kb(d.get("Buffers")),
        "shmem_mb": kb(d.get("Shmem")),
        "swap_total_mb": kb(sw_t), "swap_used_mb": kb(sw_t - sw_f),
        "pct": round(used * 100.0 / tot, 1),
    }


# ---------------------------------------------------------------- NPU / GPU
def _hint(path, driver_loaded):
    """读不到时，区分「驱动没加载」和「读不到」——两者该说不一样的话。

    ★不能用 os.path.exists(path) 判断：/sys/kernel/debug 本身是 0700 root，
      非 root 连【目录】都进不去，exists() 对底下任何文件都返回 False，
      于是「驱动没加载」这个结论会被扣到「权限不足」头上（实测踩过）。
      所以只能拿 debugfs 之外的节点当代理（devfreq / DRM 节点）。"""
    if not driver_loaded:
        return "驱动未加载"
    if not os.path.exists(path):
        return "计数器在 debugfs 里，需 root 才能读"
    return "读不到（权限或驱动状态）"


def npu():
    """VIP9000。负载来自 debugfs（VIP Lite 驱动），频率来自 devfreq（无需权限）。"""
    r = {"load": None, "cores": None, "freq_mhz": None, "max_mhz": None,
         "note": None}
    try:
        for e in os.listdir("/sys/class/devfreq"):
            p = os.path.join("/sys/class/devfreq", e)
            nm = (_read(os.path.join(p, "name")) or "").strip()
            if "npu" in nm or "vip" in nm:
                f = _read_int(os.path.join(p, "cur_freq"))
                fm = _read_int(os.path.join(p, "max_freq"))
                r["freq_mhz"] = None if f is None else round(f / 1e6)
                r["max_mhz"] = None if fm is None else round(fm / 1e6)
                break
    except OSError:
        pass
    txt = _read(_VIP_LOAD)
    if txt:
        vals = [int(x) for x in re.findall(r"Core\d+\s*:\s*(\d+)\s*%", txt)]
        if vals:
            r["cores"] = vals
            r["load"] = max(vals)
    if r["load"] is None:
        r["note"] = _hint(_VIP_LOAD, r["freq_mhz"] is not None)
    return r


def gpu():
    """PowerVR BXM-4-64。利用率只在 debugfs 的 pvr/status 里。"""
    r = {"load": None, "blocks": None, "note": None}
    txt = _read(_PVR_STATUS)
    if txt:
        m = re.search(r"GPU Utilisation:\s*(\d+)\s*%", txt)
        if m:
            r["load"] = int(m.group(1))
        blk = re.findall(r"^\s*([A-Za-z][\w]*)\s*:\s*(\d+)\s*%\s*$", txt, re.M)
        if blk:
            r["blocks"] = [{"name": k, "pct": int(v)} for k, v in blk]
    if r["load"] is None:
        r["note"] = _hint(_PVR_STATUS, os.path.exists("/sys/class/drm/renderD128"))
    return r


# ---------------------------------------------------------------- 其它
def misc():
    o = {}
    try:
        l1, l5, l15 = os.getloadavg()
        o["load"] = [round(l1, 2), round(l5, 2), round(l15, 2)]
    except OSError:
        o["load"] = None
    up = _read("/proc/uptime")
    if up:
        try:
            o["uptime_s"] = int(float(up.split()[0]))
        except (ValueError, IndexError):
            o["uptime_s"] = None
    else:
        o["uptime_s"] = None
    try:
        st = os.statvfs("/")
        tot = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        o["disk"] = {"total_gb": round(tot / 1e9, 1),
                     "used_gb": round((tot - free) / 1e9, 1),
                     "pct": round((tot - free) * 100.0 / tot, 1) if tot else 0}
    except OSError:
        o["disk"] = None
    o["soc"] = (_read("/proc/device-tree/model") or "").strip("\x00 \n") or None
    return o


# ---------------------------------------------------------------- 采样器
class Sampler:
    """一个后台线程，1 Hz 刷新快照。snapshot() 永远返回上一次的完整字典。"""

    def __init__(self, interval=1.0):
        self.interval = interval
        self._lock = threading.Lock()
        self._snap = {"t": 0, "ready": False}
        self._prev = None
        self._th = None
        self._stop = False
        self._cpus = []
        self._models = None

    def _sample(self):
        now = _cpu_times()
        util = cpu_stat(self._prev, now) if self._prev else {}
        self._prev = now
        if self._models is None:                 # 核型号/簇拓扑重启也不会变，缓存
            self._cpus = cpus()
            self._models = True
        cs = []
        for c in self._cpus:
            d = dict(c)
            d["pct"] = round(util.get(c["cpu"], 0.0), 1) if util else None
            cs.append(d)
        allu = util.get("all")
        t = temps()
        return {
            "t": time.time(),
            "ready": bool(util),
            "temps": t,
            "temp_max": t[0] if t else None,
            "cpus": cs,
            "cpu_pct": None if allu is None else round(allu, 1),
            "mem": mem(),
            "npu": npu(),
            "gpu": gpu(),
            "misc": misc(),
        }

    def _loop(self):
        while not self._stop:
            try:
                s = self._sample()
                with self._lock:
                    self._snap = s
            except Exception as e:                     # 采集永不能把服务带下去
                print("[sysmon] 采样失败:", e, flush=True)
            time.sleep(self.interval)

    def start(self):
        if self._th:
            return
        self._th = threading.Thread(target=self._loop, daemon=True,
                                    name="sysmon")
        self._th.start()

    def snapshot(self):
        with self._lock:
            return self._snap
