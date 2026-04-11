---
name: makeorder
description: 检查 backlog.json 待办任务 + 实际 git 改动，判断哪些功能已实现，更新 backlog.json 状态、写入 progress.md，并 git commit。使用场景："/makeorder" 或"整理 backlog"。
---

# Make Order

结合 `tasks/backlog.json` 待办任务和当前 git 工作区实际改动，判断已完成的工作，更新跟踪文件并提交。

## 步骤（按顺序执行，不得跳过）

### 1. 读取任务列表 + 扫描实际改动

并行执行：
- 读取 `tasks/backlog.json`，列出所有 `status` 为 `"todo"` 或 `"in_progress"` 的任务
- 运行 `git diff HEAD --stat` 和 `git status --short`，列出所有已修改/新增的文件

### 2. 综合分析已完成的工作

**两个来源都要检查：**

**来源 A — backlog 待办任务：** 对每个 todo/in_progress 任务，根据 `title`/`summary` 用 Grep/Read 确认核心逻辑是否已落地，记录 `已实现` / `未实现` / `部分实现`。

**来源 B — 实际改动文件：** 对 git diff 中涉及的改动文件，通过 `git diff HEAD -- <file>` 阅读 diff 内容，理解改动的功能意图。若改动对应 backlog 中某项任务，标记该任务已实现；若改动属于 backlog 之外的新功能或修复，单独记录下来，写入 progress.md。

### 3. 更新 backlog.json

对判断为 **已实现** 的任务：
- 将 `"status"` 改为 `"done"`
- 从原 `todo` / `in_progress` 列表中移除，添加到 `done` 列表头部
- 若原 `summary` 为空或过于简略，根据代码实际情况补充一句话描述

对 **部分实现** 或 **未实现** 的任务不修改（保留在原列表）。

### 4. 写入 progress.md

在 `tasks/progress.md` 末尾追加当天日期的记录（若当天已有记录则追加到该日期块末尾）：

```
## <YYYY-MM-DD>
- 今天做了什么：<综合描述 backlog 归档项 + backlog 之外的实际改动>
- 仍待处理：<未完成的任务 id 及原因，或"无">
- 下一个 session 启动时先看什么：<最高优先级的待办，或"todo 已清空">
```

### 5. Git commit

Stage 所有改动：

```bash
git add .
```

根据本次改动内容生成简洁的 commit message，准确描述实际做了什么（不要只写"整理 backlog"）：

- 有业务改动：用改动内容写 message，例如 `fix: 修复 read_memory 跨会话召回逻辑`
- 仅文档整理：`docs: 更新 progress.md，归档已完成任务`

使用 HEREDOC 格式提交，附上 Co-Authored-By 行。

## 注意事项

- 不要凭感觉判断"应该已实现"，必须通过 Grep/Read 或 git diff 确认
- 如果任务描述模糊，读 `tasks/progress.md` 的历史记录获取更多上下文
- 今天的日期从系统上下文（`currentDate`）读取，不要猜测
- backlog.json 不一定记录了所有改动，git diff 是最终事实来源
