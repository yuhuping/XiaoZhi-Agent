---
name: makeorder
description: 检查 backlog.json 中的 todo/in_progress 任务，判断哪些已在代码中实现，更新 backlog.json 状态、写入 progress.md，并 git commit。使用场景："/makeorder" 或"整理 backlog"。
---

# Make Order

检查 `tasks/backlog.json` 里 `todo` 和 `in_progress` 列表中的任务，判断哪些已在代码中实现，更新跟踪文件并提交。

## 步骤（按顺序执行，不得跳过）

### 1. 读取任务列表

读取 `tasks/backlog.json`，列出所有 `status` 为 `"todo"` 或 `"in_progress"` 的任务。

若 `todo` 和 `in_progress` 均为空，直接告知用户"当前没有待办任务"，结束。

### 2. 逐项核查实现状态

对每个待办任务：

- 根据 `title` 和 `summary` 推断应该修改哪些文件/类/函数
- 用 Grep / Read 查看相关代码，判断功能是否已落地
- 判断标准：核心逻辑存在于代码中，且无明显占位符或 TODO 注释指向该功能未完成

记录每项的判断结果：`已实现` / `未实现` / `部分实现`。

### 3. 更新 backlog.json

对判断为 **已实现** 的任务：
- 将 `"status"` 改为 `"done"`
- 从原 `todo` / `in_progress` 列表中移除，添加到 `done` 列表头部
- 若原 `summary` 为空或过于简略，根据代码实际情况补充一句话描述

对 **部分实现** 或 **未实现** 的任务不修改（保留在原列表）。

### 4. 写入 progress.md

在 `tasks/progress.md` 末尾追加当天日期的记录（若当天已有记录则追加到该日期块末尾）：

```
## <YYYY-MM-DD>（makeorder）
- 已完成并归档：<任务 id1>、<任务 id2>（共 N 项）
- 仍待处理：<任务 id>（原因简述）
- 下一个 session 启动时先看什么：<最高优先级的 todo 任务 id，或"todo 已清空">
```

若本次没有发现任何已实现的任务，也写一条记录说明检查结果。

### 5. Git commit

Stage 所有改动：

```bash
git add .
```

根据本次归档的任务数量和内容，生成简洁的 commit message：

- 归档了任务：`tasks: 归档已完成任务 <id1>、<id2>，更新 progress`
- 没有变化：`tasks: makeorder 检查无变更`

使用 HEREDOC 格式提交，附上 Co-Authored-By 行。

## 注意事项

- 不要修改任何业务代码，只操作 `tasks/` 下的两个文件
- 不要凭感觉判断"应该已实现"，必须通过 Grep/Read 确认代码存在
- 如果任务描述模糊，读 `tasks/progress.md` 的历史记录获取更多上下文
- 今天的日期从系统上下文（`currentDate`）读取，不要猜测
