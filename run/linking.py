# 특정 링크(A219BS010618)가 어떤 노드에서 어떤 노드로 연결되는지 확인하는 코드
import json

with open('morai_gym/lib/core/birdiview/map/link_set.json') as f:
    link_data = json.load(f)

for link in link_data:
    if link['idx'] == 'A219BS010618':
        print(f"idx: {link['idx']}")
        print(f"from_node: {link.get('from_node_idx')}")
        print(f"to_node: {link.get('to_node_idx')}")