#!/usr/bin/env python3
#import numpy as np
dir = "/home/rl/tests/digitsNew/"
def openfile(filename):
  filename = dir+filename
  print("Opening:",filename)
  with open(filename) as f:
    lines = f.readlines()
  out = []
  for line in lines:
    #print(line)
    item = line.split("[INFO]")[1]
    out.append(item[0:-1])
  #for l in out: print(l)
  return out
def compare(file1,file2):
  f1 = openfile(file1)
  f2 = openfile(file2)
  s1 = len(f1)
  s2 = len(f2)
  print(file1,":",s1,file2,":",s2)
  s = s1
  if s1 > s2: s = s2
  for i in range(s):
    if f1[i] != f2[i]:
      print(i,f1[i])
      print(i,f2[i])
if __name__ == "__main__":
 #compare("outv6","outv6_32")
 compare("outv6","outv7_32")
