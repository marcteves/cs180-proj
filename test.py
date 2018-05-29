from itertools import chain, combinations
i = set([0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13])
s = []
for z in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):
    s.append(z)

for l in s:
    print l