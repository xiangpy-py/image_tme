#!/usr/bin/env bash
# ============================================================================
# 训练结束自动关机脚本
#
# 用途：守候正在运行的训练进程，待其退出后再等 N 秒（默认 60 秒）关机。
#       针对「训练已经在前台跑着，希望跑完自动关机」的场景设计：
#       进程已经启动，无法再给它挂 && shutdown，只能由独立守护进程从旁监视。
#
# 用法：
#   # 方式一：按 PID 精确等待（推荐，最稳；PID 从 ps -ef | grep train.sh 获取）
#   nohup bash wait_and_shutdown.sh --pid 218399 > logs/console/autoshutdown.log 2>&1 &
#
#   # 方式二：按进程名匹配（模式务必写成 [t]rain.sh，否则会匹配到本脚本自身）
#   nohup bash wait_and_shutdown.sh > logs/console/autoshutdown.log 2>&1 &
#
# 取消值守：
#   pkill -f wait_and_shutdown.sh
#
# 参数：
#   --pid PID        精确等待指定 PID 退出（优先级高于 --pattern）
#   --pattern REGEX  进程匹配正则，默认 "[t]rain\.sh"
#   --delay SECONDS  进程退出后到关机之间的等待秒数，默认 60
#   --dry-run        只演练不真正关机（打印将要执行的命令），用于验证脚本逻辑
# ============================================================================
set -euo pipefail

# ---- 默认参数 ----
PATTERN='[t]rain\.sh'   # 方括号写法：本脚本命令行里含 "[t]rain.sh" 字面量，
                        # 不会与正则 [t]rain\.sh 匹配，从而避免自己匹配到自己
TARGET_PID=''
DELAY=60                # 关机前等待秒数：给文件系统回写 checkpoint 留缓冲
POLL_INTERVAL=30        # 轮询间隔（秒）：训练动辄数小时，30 秒探测足够且开销可忽略
DRY_RUN=0

# print_usage(): 打印用法说明 -> None
print_usage() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

# log_text(): 带时间戳输出一行日志 -> None
log_text() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

# is_training_alive(): 判断目标训练进程是否仍在运行 -> 0 存活 / 1 已退出
is_training_alive() {
    if [[ -n "$TARGET_PID" ]]; then
        kill -0 "$TARGET_PID" 2>/dev/null
    else
        pgrep -f "$PATTERN" >/dev/null 2>&1
    fi
}

# ------------------------------------------------------------------ #
# 参数解析
# ------------------------------------------------------------------ #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid)      TARGET_PID="$2"; shift 2 ;;
        --pattern)  PATTERN="$2";    shift 2 ;;
        --delay)    DELAY="$2";      shift 2 ;;
        --dry-run)  DRY_RUN=1;       shift ;;
        -h|--help)  print_usage; exit 0 ;;
        *)          echo "未知参数: $1（用 --help 查看用法）" >&2; exit 2 ;;
    esac
done

# ------------------------------------------------------------------ #
# 前置检查：目标进程必须正在运行
# ------------------------------------------------------------------ #
# 不做这一步的话，若训练早已结束（或 PID/模式写错），脚本会"启动即判定训练完成"
# 并直接把机器关掉，属于危险误操作。
if ! is_training_alive; then
    log_text "未检测到训练进程（pid=${TARGET_PID:-未指定}, pattern=${PATTERN}），不做任何操作并退出"
    exit 1
fi

if [[ -n "$TARGET_PID" ]]; then
    log_text "已锁定训练进程 PID=${TARGET_PID}，开始守候（轮询间隔 ${POLL_INTERVAL}s）"
else
    log_text "已锁定匹配 \"${PATTERN}\" 的训练进程，开始守候（轮询间隔 ${POLL_INTERVAL}s）"
fi

# ------------------------------------------------------------------ #
# 守候训练结束
# ------------------------------------------------------------------ #
while is_training_alive; do
    sleep "$POLL_INTERVAL"
done
log_text "训练进程已退出"

# ------------------------------------------------------------------ #
# 缓冲等待后关机
# ------------------------------------------------------------------ #
log_text "等待 ${DELAY} 秒后关机..."
sleep "$DELAY"

if [[ "$DRY_RUN" -eq 1 ]]; then
    log_text "[dry-run] 已跳过关机命令"
    exit 0
fi

log_text "执行关机"
sync                          # 刷盘：确保 kernel page cache 中的产物落盘
if ! shutdown -h now; then    # 部分容器环境 shutdown 不可用，退化为 poweroff
    log_text "shutdown 执行失败，尝试 poweroff"
    poweroff
fi
