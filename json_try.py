import json



with open("json/all.json", "r") as read_file:
    data = json.load(read_file)



dic = {}
l = []



for i in range(len(data)):
    a = data[i]["comment"]
    b = data[i]["label"]
    dic[a] = b
    if b not in l:
        print(a)
        print(b)
        l.append(b)
