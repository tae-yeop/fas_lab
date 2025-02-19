import json

with open("/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/metadata.jsonl", "r") as f:
    data1 = [json.loads(line) for line in f]

with open("/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/uhdm/metadata.jsonl", "r") as f:
    data2 = [json.loads(line) for line in f]


print(data1[:5])
print(data2[:5])
merged_data = data1 + data2

with open("merged_metadata.jsonl", "w") as f:
    for item in merged_data:
        f.write(json.dumps(item) + "\n")