import re
import subprocess

from matplotlib import animation
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("dark_background")
sns.set_palette("coolwarm")

CAP_RATIO = 0.9  # 90% cap for the GPU memory usage line

def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        used, total = map(int, result.strip().split(','))
        return used, total
    except Exception as e:
        print("Error reading GPU memory:", e)
        return 0, 1

fig, ax = plt.subplots()
bar = ax.barh(["GPU Memory"], [0], color="deepskyblue")

cap_line = None # Placeholder for potential future use

def update(frame):
    used, total = get_gpu_memory()
    bar[0].set_width(used)

    # Change bar color based on usage
    usage_ratio = used / total
    if usage_ratio < 0.6:
        bar[0].set_color("limegreen")
    elif usage_ratio < 0.85:
        bar[0].set_color("gold")
    else:
        bar[0].set_color("red")

    ax.set_xlim(0, total)
    ax.set_ylabel("Memory (MiB)")
    ax.set_title(f"GPU Memory Usage: {used} MiB / {total} MiB")

    global cap_line
    if cap_line is None:
        cap_value = int(total * CAP_RATIO)
        cap_line = ax.axvline(cap_value, color='magenta', linestyle='--', linewidth=1.5, label='90% Cap')
        ax.legend(loc="lower right")

ani = animation.FuncAnimation(fig, update, interval=1000)
plt.tight_layout()
plt.show()
