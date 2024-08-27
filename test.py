ans = []
def f(res,map):
    if len(res) == length:
        ans.append(res.copy())
        return
    for i in map.keys():
        if map[i] > 0:
            res.append(i)
            map[i] -= 1
            f(res,map)
            res.pop()
            map[i] += 1
    return

lst = [1,1,2,2]
length = 4
map1 = {}
for i in lst:
    if i in map1:
        map1[i] += 1
    else:
        map1[i] = 1

f([],map1)
print(ans)