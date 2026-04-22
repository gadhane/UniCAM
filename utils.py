import sys
import time

import torch

TERM_WIDTH = 80
TOTAL_BAR_LENGTH = 40
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time

    if total <= 0:
        raise ValueError("total must be greater than 0")

    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = max(TOTAL_BAR_LENGTH - cur_len - 1, 0)

    sys.stdout.write(" [")
    sys.stdout.write("=" * cur_len)
    sys.stdout.write(">")
    sys.stdout.write("." * rest_len)
    sys.stdout.write("]")

    cur_time = time.time()
    last_time = cur_time

    status = f" | {msg}" if msg else ""
    sys.stdout.write(status)

    padding = max(TERM_WIDTH - TOTAL_BAR_LENGTH - len(status) - 3, 0)
    sys.stdout.write(" " * padding)

    for _ in range(max(TERM_WIDTH - int(TOTAL_BAR_LENGTH / 2) + 2, 0)):
        sys.stdout.write("\b")
    sys.stdout.write(f" {current + 1}/{total} ")

    sys.stdout.write("\n" if current >= total - 1 else "\r")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds -= days * 3600 * 24
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    secondsf = int(seconds)
    seconds -= secondsf
    millis = int(seconds * 1000)

    parts = []
    if days > 0:
        parts.append(f"{days}D")
    if hours > 0 and len(parts) < 2:
        parts.append(f"{hours}h")
    if minutes > 0 and len(parts) < 2:
        parts.append(f"{minutes}m")
    if secondsf > 0 and len(parts) < 2:
        parts.append(f"{secondsf}s")
    if millis > 0 and len(parts) < 2:
        parts.append(f"{millis}ms")

    return "".join(parts) if parts else "0ms"


class NormalizeLayer(torch.nn.Module):
    """Channel-wise normalization that follows the input tensor device."""

    def __init__(self, means, sds):
        super().__init__()
        means_tensor = torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1)
        sds_tensor = torch.tensor(sds, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("means", means_tensor)
        self.register_buffer("sds", sds_tensor)

    def forward(self, input_tensor: torch.Tensor):
        return (input_tensor - self.means) / self.sds
