# BandWriting（端侧手写识别 / VelaOS QuickApp / QuickJS）

一个运行在 **VelaOS QuickApp** 下的轻量级手写字符识别 Demo，支持：
- 数字 `0-9`
- 小写字母 `a-z`
- 共 36 类

模型为自训练 TinyCNN（LeNet 风格）+ INT8 权重导出，端上 **纯 JS 推理**，不依赖 TensorFlow / ONNX 等框架。

---

## 目录结构

- `src/`：QuickApp 应用源码  
  - `src/pages/index/index.ux`：主页面（采样 + 推理）
- `train/`：训练脚本与模型产物  
  - `train/train_tiny36_28_int8_array.py`：28×28 训练 + 导出
  - `train/train_tiny36_14_int8_array.py`：14×14 训练 + 导出（推荐上设备）
  - `train/out/`：导出的权重与 json  
- `dist/`：打包产物（`.rpk`）
- `sign/`：签名文件（证书/私钥等）

---

## 模型与预处理约定（非常重要）

### 数据集
- 使用 EMNIST `byclass`
- 过滤得到 36 类：`0-9` + `a-z`

### 镜像
实际设备输入存在左右镜像问题，因此端上推理前做 **水平翻转**（镜像校正）。

### 输入尺寸
- 采样阶段固定使用 **28×28** 点阵（便于轨迹映射、调试）
- 推理阶段可选：
  - 直接 28×28 模型（更慢）
  - 先 28→14 下采样，再跑 14×14 模型（更快，推荐）

当前仓库默认推荐 14×14 模型。

---

## 训练与导出

进入 `train/`：

```bash
cd train
python train_tiny36_14_int8_array.py
````

训练完成后产物会输出到：

* `train/out/tiny36_14_best_float.pth`
* `train/out/tiny36_14_int8_array.json`

json 大小约：

* 14×14：≈ 138 KB
* 28×28：≈ 600 KB

---

## 端上推理性能

设备：REDMI Watch 5(VelaOS) + QuickJS（字节码）

当前（14×14 模型 + 28→14 下采样）单字符耗时：

* 约 **2000ms**（不同设备略有浮动）

已完成：

* buffer 预分配复用（避免 OOM / GC 抖动）
* conv2 fast path（中心区域无边界判断）
* 权重 typed array 化（bias Float32Array / w_q Int16Array / conv2 wf Float32Array）

---

## 已知问题

* 小写 `a` 容易与 `6` 混淆（14×14 下采样会损失细节）



---

## 自动识别

支持“停笔 3 秒自动识别”：

* `onUp()` 后启动 3s timer
* 3s 内无继续输入则触发自动推理
* 自动推理可配置是否展示预览（会增加内存与 UI 开销）

---

## 打包与运行

项目已包含打包产物：

* `dist/moe.orpu.bandwriting.release.1.0.0.rpk`

具体打包/安装流程依赖 VelaOS QuickApp 工具链与设备环境，这里不展开。

---

## 商业化与许可

### 许可证

本项目代码与模型权重文件均采用 **MIT License** 开源。
 
要求：

* 保留原始版权声明与许可证文本

---

### 模型权重说明

本仓库中的模型权重文件（`tiny36_*.json` / `.pth`）：

* 由本项目自行训练生成
* 不包含第三方受限模型
* 不包含商业授权字体
* 不依赖闭源 AI 框架推理

---

### 关于数据集

模型训练使用 EMNIST 数据集。

EMNIST 基于 NIST Special Database 19 派生，属于公开研究数据集。

在商业使用前，建议确认：

* 当前使用的 EMNIST 版本的具体再分发条款
* 是否存在额外署名或引用要求

本项目不附带原始数据集，仅包含训练脚本。

---

### 免责声明

本项目为研究与工程实践示例，不对任何实际应用场景的识别准确率、适用性或法律合规性作保证。

商业化使用者应自行确认数据来源合规性与应用场景合法性。

---
