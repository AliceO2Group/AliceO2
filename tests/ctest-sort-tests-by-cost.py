#!/usr/bin/env python3

tests=[]

with open('Testing/Temporary/CTestCostData.txt','r') as reader:
    for line in reader:
        r = line.split(' ')
        if len(r) != 3:
            break
        tests += [ ('{:7.3f}'.format(float(r[2])),r[0]) ]

tests.sort(key=lambda x: x[0], reverse=True)

print(*tests,sep='\n')
