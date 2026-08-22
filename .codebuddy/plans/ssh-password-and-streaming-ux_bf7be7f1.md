---
name: ssh-password-and-streaming-ux
overview: 为模型转换与 VM SSH 测试补齐一次性密码认证入口，并修复 SSH 无换行交互提示在面板中不可见、无法可靠引导输入的问题。
design:
  architecture:
    framework: html
  styleKeywords:
    - 深色工具面板
    - 敏感信息保护
    - 实时交互反馈
    - 清晰状态引导
  fontSystem:
    fontFamily: PingFang SC
    heading:
      size: 26px
      weight: 700
    subheading:
      size: 16px
      weight: 600
    body:
      size: 14px
      weight: 400
  colorSystem:
    primary:
      - "#3B82F6"
      - "#60A5FA"
    background:
      - "#111827"
      - "#1F2937"
    text:
      - "#F9FAFB"
      - "#94A3B8"
    functional:
      - "#F59E0B"
      - "#22C55E"
      - "#EF4444"
todos:
  - id: trace-ssh-flow
    content: 使用 [subagent:code-explorer] 核对 SSH 输入、日志与配置持久化调用链
    status: completed
  - id: add-session-password
    content: 在 train_panel.py 增加不持久化的 VM 密码输入与任务认证上下文
    status: completed
    dependencies:
      - trace-ssh-flow
  - id: stream-ssh-prompts
    content: 改造 train_panel.py 日志流，实时展示无换行 SSH 交互提示
    status: completed
    dependencies:
      - add-session-password
  - id: route-auth-input
    content: 在两脚本中实现受控密码转发、提示识别与标准输入继承
    status: completed
    dependencies:
      - stream-ssh-prompts
  - id: verify-security-flow
    content: 验证密钥认证、指纹确认、密码认证、脱敏与任务清理流程
    status: completed
    dependencies:
      - route-auth-input
---

<h2>用户需求</h2>
<p>优化模型转换与“测试 VM SSH”的认证体验：面板需支持填写 SSH 密码，但密码仅限当前任务使用，不写入默认配置、配置文件、命令行、复制命令或日志。</p>
<h2>产品概述</h2>
<p>模型转换页在保留 SSH 密钥免密连接的同时，为需密码认证的虚拟机提供清晰、安全、可操作的交互流程。</p>
<h2>核心功能</h2>
<ul>
<li>模型转换区域新增 VM Password 密码输入，默认隐藏且不持久化。</li>
<li>实时显示无换行的 SSH 指纹确认与密码提示，避免日志看似卡死。</li>
<li>识别确认与密码提示，明确引导发送 <code>yes</code> 或密码。</li>
<li>密码仅在匹配密码提示时转发，日志始终脱敏。</li>
<li>保留现有手动日志输入、停止任务与 SSH 密钥认证行为。</li>
</ul>

<h2>技术栈选择</h2>
<ul>
<li>沿用现有 Python 标准库 HTTP 面板、内嵌 HTML/CSS/JavaScript 与 Windows OpenSSH 调用方式。</li>
<li>不新增第三方依赖，不将凭据交给命令行参数或默认配置。</li>
</ul>
<h2>实施方案</h2>
<p>在 <code>train_panel.py</code> 中将 VM 密码作为仅存在于浏览器当前页面与单次启动请求中的敏感字段；服务端将其与可持久化表单值分离，任务结束立即清理。日志读取改为“原始输出增量展示 + 完整行标记解析”双通道：短时间缓冲的无换行内容及时展示，只有完整行才交给现有训练进度与标记解析，避免逐字写入造成日志数组膨胀和频繁页面刷新。</p>
<p>在 <code>host_train_export.py</code> 明确 SSH/SCP 子进程的标准输入继承链，确保面板写入的任务标准输入可传递给当前 SSH 命令。面板根据输出识别主机指纹确认与密码提示；只有检测到密码提示且本次任务存在临时密码时，才一次性发送密码。未填写密码时保持现有人工输入流程，兼容密钥认证。</p>
<h2>实施注意事项</h2>
<ul>
<li>密码字段不得加入前端 <code>fields</code> 持久化列表、默认值对象、示例 JSON、状态响应、日志或复制命令。</li>
<li>任务启动、取消、异常退出和正常退出均清理内存中的临时密码及提示状态。</li>
<li>交互提示只触发一次对应的自动密码投递，防止重复输出或连续 SSH/SCP 阶段误投递。</li>
<li>普通手动输入保留回显；密码输入统一记录为脱敏状态，不输出长度或内容。</li>
<li>不改动训练、导出和非 SSH 命令的执行路径，控制改动范围。</li>
</ul>
<h2>架构设计</h2>
<ul>
<li><strong>转换表单：</strong>VM 密码仅作为敏感会话输入提交。</li>
<li><strong>任务控制：</strong>保存非敏感任务参数与独立的临时认证上下文。</li>
<li><strong>进程输出：</strong>增量日志展示器显示交互提示；行解析器继续处理既有状态标记。</li>
<li><strong>认证转发：</strong>提示识别器决定是否使用临时密码或等待用户通过日志输入框提交。</li>
</ul>
<h2>目录结构</h2>
<pre>
E:/vscode_workspace/myAUTOtrain/
├── train_panel.py                     # [修改] 增加 VM 密码会话字段、交互提示展示/识别、受控密码转发及转换页输入控件。
└── host_train_export.py               # [修改] 明确 SSH/SCP 子进程标准输入继承，保证面板交互输入能传递到当前 SSH 阶段。
</pre>

<h2>界面设计</h2>
<p>在现有深色模型转换卡片中，将 VM Password 放在 VM User 与 VM Host/IP 相邻位置，使用密码掩码、显示/隐藏控制和“仅本次任务使用，不会保存”辅助说明。日志页保留现有输入框，在检测到 SSH 提示时显示醒目的状态说明，并将输入模式切换为“确认输入”或“密码输入”；密码模式保持掩码与脱敏反馈。</p>

# Agent Extensions

<h2>Agent Extensions</h2>
<h3>SubAgent</h3>
<ul>
<li><strong>code-explorer</strong>：核对任务启动、状态序列化、默认保存、日志轮询与 SSH/SCP 调用链。预期产出：确认敏感字段隔离点及无回归修改范围。</li>
</ul>