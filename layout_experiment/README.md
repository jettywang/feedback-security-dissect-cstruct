# dissect.cstruct 布局实验：指针宽度、对齐与匿名 union 的最小复现

针对“32 位固件样本解析正常、64 位样本字段错位”的问题，本目录用一个含
**指针 + 匿名 union + 嵌套数组**的最小结构，把三个影响因素拆开测量，并给出
可运行复现、手工字节图和目标 ABI 配置建议。

## 被测结构

```c
struct Inner { uint16 id; uint16 flags; };   // 4 字节，自然对齐 2（成员最大对齐）

struct Rec {
    uint32 magic;
    void  *next;                 // 指针：ILP32=4 字节 / LP64=8 字节
    union {                     // 匿名 union，尺寸=max(成员)=4，对齐 4
        uint32       type;
        struct Inner info;
        uint32       raw;
    };
    uint16 count;
};

struct Block { Rec item[2]; };  // 嵌套数组，专门验证第二个元素的偏移
```

同一定义用三种策略建模：

- `packed`：`cs.load(..., align=False)`，无任何填充，尺寸即字段宽度之和。
- `natural`：`cs.load(..., align=True)`，按目标 ABI 自然对齐并做尾部填充。
- `explicit`：仍 `align=False`，但定义里手写 `char padNN[n]` 保留字段，
  字节级复刻自然布局，不依赖对齐器。

## 如何运行

```bash
# 主复现：布局表 + 字节图 + 引擎/cstruct 交叉核对 + 第二元素 + 主机陷阱
.venv/bin/python layout_experiment/run_experiment.py

# 真实 C 编译器对照（ILP32 用 4 字节整数模拟指针，无需 32 位 sysroot）
.venv/bin/python layout_experiment/compare_c_reference.py

# 固化全部结论的测试（35 项，含编译/解释两条执行路径）
.venv/bin/pytest layout_experiment/ -q
```

## 布局表（offset / size，已与 C 编译器和 dissect.cstruct 三方核对）

| 目标 | 策略 | sizeof(Rec) | sizeof(Block) | magic | next | 匿名 union | count |
|------|------|------------:|--------------:|------:|-----:|-----------:|------:|
| ILP32 | packed   | 14 | 28 | 0 | 4 | 8  | 12 |
| ILP32 | natural  | 16 | 32 | 0 | 4 | 8  | 12 |
| ILP32 | explicit | 16 | 32 | 0 | 4 | 8  | 12 |
| LP64  | packed   | 18 | 36 | 0 | 4 | 12 | 16 |
| LP64  | natural  | 24 | 48 | 0 | 8 | 16 | 20 |
| LP64  | explicit | 24 | 48 | 0 | 8 | 16 | 20 |

LP64 natural 的 `Rec` 含两段填充：`next` 前 4 字节（offset 4–7）和 `count`
后 2 字节尾填充（offset 22–23）。ILP32 natural 只有 2 字节尾填充
（offset 14–15）。

### 嵌套数组第二个元素 `item[1]`

数组步长等于**带尾部填充的元素尺寸**，因此第二元素起点即 `sizeof(Rec)`：

| 目标 | 策略 | item[1] 起点 | item[1].count 绝对偏移 |
|------|------|-------------:|-----------------------:|
| ILP32 | packed   | 14 (0x0e) | 26 (0x1a) |
| ILP32 | natural  | 16 (0x10) | 28 (0x1c) |
| LP64  | packed   | 18 (0x12) | 34 (0x22) |
| LP64  | natural  | 24 (0x18) | 44 (0x2c) |

`run_experiment.py` 第 4 节用按字节偏移生成的魔数缓冲区，逐字段证明第二元素
从上述偏移读出，且 union 重叠成员（`type`/`raw`/`info.*`）读到同一批字节。

## 三个因素各自的影响（隔离测量）

1. **指针宽度（packed 下 32→64，唯一变量）**：`next` 4→8 字节，其后的 union
   与 `count` 整体后移 4 字节（union 8→12，count 12→16）。
2. **对齐（64 位下 packed→natural，指针宽度不变）**：`next` 因 8 字节对齐
   从 offset 4 推到 8（插入 4 字节前填充），union 12→16、count 16→20，
   结构再按 8 对齐做 2 字节尾填充，Rec 18→24。
3. **union 尺寸**：本例 union 为 `max(uint32, Inner=4, uint32)=4`，两种 ABI
   下都是 4 字节、对齐 4，本身不随位宽变化；它的影响是其 **4 字节对齐要求**
   决定了 `count` 的合法起点，以及 packed 下 union 对齐降为 1 的差异。

## 为什么不能用主机 ctypes 的默认布局

`dissect.cstruct.cstruct()` 的默认指针类型取自主机 `sys.maxsize`
（见 `dissect/cstruct/cstruct.py`：`uint64 if sys.maxsize > 2**32 else uint32`）。
在 64 位分析主机上解析 32 位固件而不显式指定指针类型时，模型会把 4 字节
指针当成 8 字节。`run_experiment.py` 第 6 节实测：

```
wrong model (host default): next@8  size=24
right model (pinned u32)  : next@4  size=16
parsed 'count': wrong=0x0          correct=0x4444
parsed 'next' : wrong=0x444433333333  correct=0x22222222
```

字段错位来自**定义/配置**（指针宽度与对齐策略），不是读取执行路径。

## compiled 与 interpreted 对照

同一 `cstruct` 实例分别以 `compiled=True/False` 加载，12 组（2 ABI × 3 策略
× 2 路径）的 offset、`sizeof(Rec)`、`sizeof(Block)` 完全一致，字节往返也一致。
因此差异只能来自结构定义或目标配置，而非编译/解释执行方式。

写回语义差异（与布局正确性无关）：natural 模型的**隐式**填充在 `dumps()` 时
输出零字节；packed/explicit 没有隐式填充，explicit 的 `padNN` 是具名字段会
原样保留。

## 目标配置建议

- 始终显式钉住目标指针宽度，不要继承主机默认：
  `cstruct(pointer="uint32")`（32 位固件）或 `cstruct(pointer="uint64")`。
- 固件结构若由 C 编译器按默认 ABI 生成，用 `cs.load(def, align=True)`；
  若头文件带 `__attribute__((packed))` / `#pragma pack(1)`，用默认
  `align=False`。两者不可混用。
- 需要跨工具/跨版本稳定或想完全掌控填充时，用 explicit 模型手写 `padNN`
  保留字段并保持 `align=False`，其字节布局与 natural 完全相同（已逐字节核对）。
- 以固件自身的 ABI/头文件为准确定位宽与对齐；分析主机的位宽只决定进程环境，
  与目标结构布局无关。本实验的所有偏移均由目标参数化引擎和真实 C 编译器
  得出，未读取主机 `ctypes` 布局。

## 文件说明

- `abi_layout.py`：目标参数化的独立布局引擎（指针宽度/对齐/union 规则显式传入）。
- `cstruct_models.py`：dissect.cstruct 的 packed/natural/explicit 三种建模。
- `byteprobe.py`：按字节偏移生成魔数缓冲区、字段读取与 `dumps()` 往返校验。
- `run_experiment.py`：端到端复现与完整报告（输出见 `sample_output.txt`）。
- `reference_layout.c`：独立 C 参考，含 `_Static_assert`，支持模拟 ILP32。
- `compare_c_reference.py`：编译四种 C 变体并与 Python 引擎逐项比对。
- `test_layout_experiment.py`：35 项 pytest，覆盖偏移、尺寸、第二元素、往返与主机陷阱。
