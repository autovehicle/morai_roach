import json
import numpy as np

with open("morai_gym/lib/core/birdiview/map/stoplane_marking_set.json") as f:
    sl_data = json.load(f)

# C119BS010063 위치 근처 정지선 찾기
tl_x, tl_y = 63.78, 1490.74

for sl in sl_data:
    pts = sl['points']
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    dist = np.sqrt((cx - tl_x)**2 + (cy - tl_y)**2)
    if dist < 50:
        print(f"idx={sl['idx']}, dist={dist:.1f}m, center=({cx:.1f}, {cy:.1f})")