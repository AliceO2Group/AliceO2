#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------
# replace the text between two lines starting with 'delimiter' in file 'fold'
# by the text between two lines starting with 'delimiter' in file 'ftouse'.
# Write new content into 'fnew' and keep the lines with the 'delimiter'.

# -----------------------------------------------------------------------------
# get text in file 'fn' beteween the lines starting with 'delimiter'
def blockbtwdelims (fn, delimiter):
  blck = []

  cnt = 0
  with open(fn) as f:
    for line in f:
      if line.startswith(delimiter):
        blck.append(line.rstrip())
        if cnt > 0:
          break
        cnt += 1
      else:
        if cnt > 0:
          blck.append(line.rstrip())

  return blck

# -----------------------------------------------------------------------------
# get text in file 'fn' before any line starting with 'delimiter'
def blockbefdelims (fn, delimiter):
  blck = []

  with open(fn) as f:
    for line in f:
      if line.startswith(delimiter):
        break
      blck.append(line.rstrip())

  return blck

# -----------------------------------------------------------------------------
# get text in file 'fn' after the text block delimited by lines starting with
# 'delimiter'
def blockaftdelims (fn, delimiter):
  blck = []

  cnt = 0
  with open(fn) as f:
    for line in f:
      if line.startswith(delimiter):
        if cnt < 2:
          cnt += 1
          continue

      if cnt > 1:
        blck.append(line.rstrip())

  return blck

# -----------------------------------------------------------------------------
# concatenate two blocks of text
def addblocks(b0, b1):
  b2 = b0
  for l in b1:
    b2.append(l.rstrip())

  return b2

# -----------------------------------------------------------------------------
def main(initCard):

  if len(sys.argv) < 4:
    print ("Wrong number of arguments!")
    print ("Usage:")
    print ("  purger.py cc fn2u fnold fnnew")
    print ("")
    print ("    cc: 1: AO2D, 2: Helpers, 3: PWGs, 4: Joins")
    print ("    fn2u: file with new text")
    print ("    fnold: file with old text")
    print ("    fnnew: file with replaced text")
    print ("")
    exit()

  cc = int(sys.argv[1])
  fntouse = sys.argv[2]
  fnold = sys.argv[3]
  fnnew = sys.argv[4]

  # get the 'delimiter' from initCard
  tmp = None
  if cc == 1:
    tmp = initCard.find("O2general/delimAO2D")
  elif cc == 2:
    tmp = initCard.find("O2general/delimHelpers")
  elif cc == 3:
    tmp = initCard.find("O2general/delimPWGs")
  elif cc == 4:
    tmp = initCard.find("O2general/delimJoins")
  else:
    exit()
  delimiter = tmp.text.strip()
  print("Replacing ",delimiter)

  # get replacement
  b2u = blockbtwdelims(fntouse, delimiter)
  if len(b2u) == 0:
    exit()

  # entire new text
  bnew = addblocks(blockbefdelims(fnold, delimiter), b2u)
  bnew = addblocks(bnew, blockaftdelims(fnold, delimiter))

  # write new text to fnnew
  with open(fnnew, 'w') as f:
    for l in bnew:
      print(l, file=f)

# -----------------------------------------------------------------------------
if __name__ == "__main__":

  initCard = ET.parse("inputCard.xml")

  main(initCard)

# -----------------------------------------------------------------------------
