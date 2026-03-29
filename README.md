# maimai-timing-align

基于 Streamlit 的舞萌音频对齐与视频重编码工具：
- Clip1（主视频）支持上传 <= 500MB
- Clip2 支持上传视频 / 上传音频 / 输入 Song ID 后从 LXNS 下载 mp3
- 对齐时自动抽取音轨并调用 `otoge-service` 的音频对齐接口
- 提供预览导出（可静音 A/B 轨）与最终导出
- 导出支持体积预设、分辨率、帧率、H.264/H.265

## 环境

- Python 3.14
- `uv`
- `ffmpeg` / `ffprobe`（开发模式下需在 PATH 中；打包版会随程序分发）
- 可访问 `otoge-service`（默认 `http://127.0.0.1:8000`）

## 安装

```bash
uv sync
```

## 启动

```bash
uv run streamlit run streamlit_app.py
```

## 使用说明

1. 上传 Clip1（主视频）
2. 选择 Clip2 来源：
   - 上传视频
   - 上传音频
   - 输入 Song ID，从 `https://assets2.lxns.net/maimai/music/{song_id}.mp3` 下载
3. 点击“开始自动对齐”
4. 在“测试预览”中选择静音 A/B 和预览时长，生成预览并拖动进度条试听
5. 满意后“导出完整视频”

## 输出规则（当前版本）

- 画面：仅使用 Clip1
- 音频：Clip1 与 Clip2 对齐后混音，可调 A/B 响度与混响强度
- 时间范围：从自动对齐起点开始，到可重叠尾段结束

## otoge-service 接口说明

- 客户端严格解析统一响应包裹：`{ code, message, data }`
- 仅使用 `data` 作为业务结果
- 默认请求路径：`/audalign/audalign/align`
- 若服务开启了开发者令牌校验，请在 UI 输入 `Developer Token`

## 常见问题

- 对齐失败：确认 `otoge-service` 可访问、URL 正确、token 正确
- 下载 Song ID 失败：检查 Song ID 是否存在，或稍后重试
- 导出失败：确认 `ffmpeg -version` 和 `ffprobe -version` 可执行
