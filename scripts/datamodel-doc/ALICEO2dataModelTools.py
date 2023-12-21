#!/usr/bin/env python3
import sys
import regex as re
import numpy as np
import nltk

# -----------------------------------------------------------------------------
# functions
#
# -----------------------------------------------------------------------------
# concatenates a list of strings
#   words: list of strings
#   withspace: words are separated by space
#   space: by default ' '

def block(words, withspace=True, space=" "):
  sep = ""
  if withspace == True:
    sep = space
  cont = ""

  if len(words) == 0:
    return cont

  if isinstance(words[0], str):
    for w in words:
      cont += w+sep
  else:
    for w in words:
      cont += w.txt+sep

  return cont

# -----------------------------------------------------------------------------
# block of words is split into list of words
#   block: string of words

def split(block):

  # split into words
  words = nltk.word_tokenize(block)

  return words

# -----------------------------------------------------------------------------
# holds a word and the corresponding line number

class word:
  def __init__(self, txt, lnr):
    self.txt = txt
    self.lnr = lnr

# .............................................................................
# holds a define

class define:
  def __init__(self, name, line):
    self.name = name
    self.vars = list()

    # how many parameters
    vars = "".join(line.split("(")[1:]).split(")")[0].split(",")
    if vars[0] != "":
      self.vars = vars
      self.cont = ")".join(line.split(")")[1:]).strip()
    else:
      self.cont = line

  def expandLine(self, line):
    expandedLine = " ".join(line.split())

    if self.name in line:
      if len(self.vars) == 0:
        # no substitution of variables needed
        expandedLine = line.replace(self.name, self.cont)
      else:
        inds = [i for i in range(len(line)) if line.startswith(self.name, i)]
        expandedLine = line[:inds[0]]
        for i, ind in enumerate(inds):

          # make sure that the name of the define == self.name and not only starts with self.name
          words = line[ind:].split('(')
          if words[0].strip() != self.name:
            if i < len(inds)-1:
              expandedLine += line[ind:inds[i+1]]
            else:
              expandedLine += line[ind:]
            continue

          # substitute variables vars
          vars = "".join(line[ind:].split("(")[1:]).split(")")[0].split(",")
          if len(vars) != len(self.vars):
            print("ATTENTION")
            print("Substitution error!")
            print('>> ', line)
          else:
            words = split(self.cont)
            for ind1 in range(len(self.vars)):
              for ind2 in range(len(words)):
                if self.vars[ind1].strip() in words[ind2]:
                  words[ind2] = re.sub(self.vars[ind1].strip(), vars[ind1].strip(), words[ind2])
            expandedLine += block(words)

      # remove ##, which connects two strings
      expandedLine = block(split(expandedLine.replace(" # # ", "")))

    return expandedLine

  def print(self):
    print("    define: "+self.name)
    print("      vars:", len(self.vars))
    for var in self.vars:
      print("        ", var)
    print("   content: "+self.cont)

# -----------------------------------------------------------------------------
# content is a tuple<list<word>, list<str>>
# create tuple with (content[0][[i1:i2],[i3:i4]], content[1])

def select(content, i1, i2=-1, i3=-1, i4=-1):
  wsel = list()

  if i2 < 0:
    i2 = len(content[0])
  for w in content[0][i1:i2]:
    wsel.append(w)

  if i3 >= 0:
    if i4 < 0:
      i4 = len(content[0])
    for w in content[0][i3:i4]:
      wsel.append(w)

  return wsel, content[1]

# -----------------------------------------------------------------------------
# compute number of open brackets at each character of a string
#   obr/cbr: openening/closing brackets
#            brackets can include more than on character
#   line: string to process

def countBrackets(obr, cbr, line):
  seq = [0]*len(line)

 # find obr
  answ = list(map(lambda x: line[x:x + len(obr)]
              == obr, range(len(line) - len(obr) + 1)))
  iop = [i for i, x in enumerate(answ) if x == True]

  if obr != cbr:
    # find cbr
    answ = list(map(lambda x: line[x:x + len(cbr)]
                == cbr, range(len(line) - len(cbr) + 1)))
    icl = [i for i, x in enumerate(answ) if x == True]

    # build sequence which holds the number of open brackets
    # at each character of the line
    for i1 in iop:
      for i2 in range(i1, len(line)):
        seq[i2] += 1
    for i1 in icl:
      for i2 in range(i1+len(cbr), len(line)):
        seq[i2] -= 1

  else:
    # build sequence which holds the number of open brackets
    # at each character of the line
    cnt = -1
    inc = 1
    for i1 in iop:
      cnt += 1
      if np.mod(cnt,2) == 0:
        inc = 1
      else:
        inc = -1
      for i2 in range(i1, len(line)):
        seq[i2] += inc

  return seq

# -----------------------------------------------------------------------------
# find text in closed brackets
# number of closing brackets matches the number of previously opening brackets
#   obr/cbr: openening/closing brackets
#            brackets can include more than one character
#   line: string
#   withheader: all text from the beginning until closing brackets match
# return begin/end position of brackets

def findInBrackets(obr, cbr, line):
  newline = ''
  seq = countBrackets(obr, cbr, line)

  # find first seq == 1 (opening)
  inds = [i for i, x in enumerate(seq) if x == 1]
  if len(inds) > 0:
    oi = inds[0]

    # find next seq == 0 (closing)
    inds = [i for i, x in enumerate(seq[oi:]) if x == 0]
    if len(inds) > 0:
      ci = inds[0]+oi
    else:
      sys.exit('<findInBrackets> '+obr+' ... '+cbr+' missmatch! EXIT -->')

  return [oi, ci]

# -----------------------------------------------------------------------------
def lineInBrackets(obr, cbr, line, withheader=False):
  newline = ''
  [oi, ci] = findInBrackets(obr, cbr, line)

  if withheader:
    newline = line[:ci]
  else:
    newline = line[oi:ci]

  return newline

# -----------------------------------------------------------------------------
# remove text from line between brackets obr..cbr
#   obr/cbr: openening/closing brackets
#            brackets can include more than on character
#   line: string
#   stat: number of not completed brackets (needs to be >= 0)
# return modified line and int stat

def removeInBrackets(obr, cbr, line, stat):
  seq = countBrackets(obr, cbr, line)
  for ii in range(len(seq)):
    seq[ii] += stat

  # compute the results
  stat = seq[-1]
  if stat < 0:
    sys.exit('<removeInBrackets> '+obr+' ... '+cbr+' missmatch! EXIT -->')

  # only select characters with seq[]=0
  newline = ""
  for i1 in [i for i, x in enumerate(seq) if x == 0]:
    newline += line[i1]

  return stat, newline

# -----------------------------------------------------------------------------
# is a contained in b?
#   a: list of strings
#   b: list of words

def list_in(a, b):

  # create list of strings
  b2u = list()
  for w in b:
    b2u.append(w.txt)

  # compare a and b2u
  return list(map(lambda x: b2u[x:x + len(a)] == a, range(len(b2u) - len(a) + 1)))

# -----------------------------------------------------------------------------
# calls of templated task structures can be complicated, e.g.
# JetMatching<
#   soa::Filtered<soa::Join<aod::Collisions, aod::McCollisionLabels>>::iterator,
#   o2::aod::MCDetectorLevelJets,
#   o2::aod::MatchedMCDetectorLevelJets,
#   o2::aod::MCParticleLevelJets,
#   o2::aod::MatchedMCParticleLevelJets
# >
#
# This has 5 arguments which need to be properly extracted.
#   words: a list of strings starting with "JetMatching......"

def getArgumentValues(words):
  line = block(words)

  # get the argument line
  argLine = lineInBrackets("<",">",line)[1:-1]

  # find further <...> in argLine
  seq = countBrackets("<", ">", argLine)
  # a ',' is only accepted as separator of arguments when it is outside of a <...>
  ci = [i for i, char in enumerate(argLine) if (char == ',' and seq[i] == 0)]

  # split the argLine according to the accepted argument separators
  argValues = list()
  i0 = 0
  for i in ci:
    arg = "".join(argLine[i0:i-1].split())
    argValues.append(arg)
    i0 = i+1
  arg = "".join(argLine[i0:].split())
  argValues.append(arg)

  return argValues

# -----------------------------------------------------------------------------
def pickContent(lines_in_file):

  # 1. remove the comments // but not the //!
  #   ATTENTION: '//' can be part of a string, e.g. http://alice-ccdb.cern.ch
  # 2. consider extensions \
  # 3. remove comment blocks /* ... */
  linesWithoutComments = list()
  lineToAdd = ""
  for line in lines_in_file:

    # 1. remove the comments // but not the //!
    l = ' '.join(line.split())+' '
    obr = countBrackets('"', '"', l)
    #print("line: ", l)
    #print(" obr: ",obr)
    i1 = l.find("//")
    while i1 >= 0:
      if obr[i1] == 0 and l[i1+2] != "!":
        l = l[0:i1].strip()
      i1 = l.find("//", i1+2)
    if l == "":
      continue

    # 2. consider extensions \
    if l.strip().endswith("\\"):
      lineToAdd = lineToAdd+" "+l[:len(l)-2].strip()
    else:
      lineToAdd = lineToAdd+" "+l
      linesWithoutComments.append(lineToAdd)
      lineToAdd = ""

  # 3. remove comment blocks /* ... */
  stat = 0
  for ind in range(len(linesWithoutComments)):
    res = removeInBrackets("/*", "*/", linesWithoutComments[ind], stat)
    stat = res[0]
    linesWithoutComments[ind] = res[1]

  # select all lines starting with #define
  idfs = [l for l, s in enumerate(linesWithoutComments) if s.lstrip().startswith("#define")]
  for idf in idfs:
    ws = split(linesWithoutComments[idf])
    defstring = linesWithoutComments[idf].split(ws[2], 1)[1]
    df = define(ws[2], defstring)

    # find the corresponding #undef
    # if no #undef then apply to the end of the file
    iend = len(linesWithoutComments)
    iudfs = [l for l, s in enumerate(linesWithoutComments) if s.lstrip().startswith("#undef")]
    for iudf in iudfs:
      ws = split(linesWithoutComments[iudf])
      if ws[2] == df.name:
        iend = iudf-1
        break

    # substitute #define within the def-undef block
    for ii in range(idf+1, iend):
      linesWithoutComments[ii] = df.expandLine(linesWithoutComments[ii])

  # create list of word(s)
  words = list()
  for ind in range(len(linesWithoutComments)):
    # for this remove the //! comments
    l2u = linesWithoutComments[ind]
    if l2u.strip() == "":
      continue
    i1 = l2u.find("//!")
    if i1 >= 0:
      l2u = l2u[0:i1].strip()
    for w in split(l2u):
      words.append(word(w, ind))

  content = (words, linesWithoutComments)

  return content

# -----------------------------------------------------------------------------
