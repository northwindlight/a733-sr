# 部署用的 systemd 叠加片段

`sr-web.service` 本体在 `deploy/` 下，这里放的是**叠加在它上面的 drop-in**。

装法（板上）：

```bash
sudo mkdir -p /etc/systemd/system/sr-web.service.d
sudo install -m644 deploy/systemd/10-root.conf \
    /etc/systemd/system/sr-web.service.d/10-root.conf
sudo systemctl daemon-reload && sudo systemctl restart sr-web
```

## 10-root.conf —— 为什么要 root

只为一件事：GPU / NPU 的硬件计数器在 debugfs 里，而 `/sys/kernel/debug`
**整个目录是 `drwx------ root`**，普通用户连目录都进不去。注意**不是文件 0600，
是父目录 0700** —— 光给文件 chmod 不管用。不 root 的话状态栏那两格只能显示 `—`。

代价是上传的文件和 ffmpeg / srpipe 子进程也以 root 跑。本服务只在内网、
只收自己人的视频，这个代价是有意接受的。想收回的话，改成普通用户跑服务
+ 一个只读这两个计数器的 root 小采集器。
