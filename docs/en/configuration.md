# Configuration (`config.ini`)

YOLOX Wrapper manages application settings through a `config.ini` file.  
You can define multiple profiles and switch between them using the GUI dropdown.

## File Format

```ini
; YOLOX Wrapper configuration file

[default]
device = cpu
model_size = l
batch_size = 16
imgsz = 640
workers = 4
val_split = 0.2
output_dir =
conf = 0.25
iou = 0.45

[factory_pc]
device = cuda:0
model_size = l
batch_size = 16
imgsz = 640
workers = 4
val_split = 0.2
output_dir =
conf = 0.25
iou = 0.45
```

## Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `device` | str | `cpu` | Device to use. `cpu` or `cuda:0`, etc. |
| `model_size` | str | `l` | Model size. `nano` / `tiny` / `s` / `m` / `l` / `x` |
| `batch_size` | int | `16` | Training batch size |
| `imgsz` | int | `640` | Input image size (square) |
| `workers` | int | `4` | DataLoader worker count |
| `val_split` | float | `0.2` | Validation data ratio (0.0–1.0) |
| `output_dir` | str | `""` | Output directory (empty uses default) |
| `conf` | float | `0.25` | Inference confidence threshold |
| `iou` | float | `0.45` | NMS IoU threshold |

## Profiles

- The `[default]` section is required and cannot be deleted.
- Add profiles with any section name (e.g., `[factory_pc]`, `[dev_pc]`).
- Switching profiles in the GUI dropdown automatically applies settings to all tabs.
- Specifying a non-existent profile falls back to `[default]`.

## Using from Code

```python
from yolox_wrapper import AppConfig, ProfileParams

config = AppConfig()          # Load config.ini
config.load()

# List profiles
print(config.profiles())     # ["default", "factory_pc", "dev_pc"]

# Get values
params = config.get("factory_pc")
print(params.device)          # "cuda:0"

# Update values
config.set("factory_pc", "batch_size", "32")
config.save()

# Add/remove profiles
config.add_profile("new_env")
config.remove_profile("dev_pc")
```
