# 모든 신호등에 대해 근처(20m 이내내) 링크 후보를 출력하는 코드드

import json
import numpy as np

with open('morai_gym/lib/core/birdiview/map/traffic_light_set.json') as f:
    tl_data = json.load(f)

with open('morai_gym/lib/core/birdiview/map/link_set.json') as f:
    link_data = json.load(f)

for tl in tl_data:
    tx, ty = tl['point'][0], tl['point'][1]
    
    nearby_links = []
    for link in link_data:
        pts = link['points']
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        d = np.sqrt((cx - tx)**2 + (cy - ty)**2)
        if d < 20:
            nearby_links.append((d, link['idx']))
    
    nearby_links.sort(key=lambda x: x[0])
    print(f"\n[{tl['idx']}] 현재={tl['link_id_list']}")
    for d, link_idx in nearby_links:
        print(f"  {link_idx} dist={d:.1f}m")