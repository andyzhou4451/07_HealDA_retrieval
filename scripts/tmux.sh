# 创建一个新会话
tmux new -s <name>

# 分离当前会话，在后台运行
Ctrl+b d 或者 tmux detach

# 列出所有正在运行的会话
tmux ls

# 重新连接到一个已有会话
tmux attach -t <name>

# 终止指定会话
tmux kill-session -t <name> 

# 从列表中选择一个会话
Ctrl+b s 或 Ctrl w

# 重命名当前会话
Ctrl+b $

# 将当前窗格垂直分割左右两个
Ctrl+b %

# 将当前窗格水平分割为上下两个
Ctrl+b \"

# 在相邻窗格之间移动焦点
Ctrl+b <方向键>

# 关闭当前窗格
Ctrl+b x

