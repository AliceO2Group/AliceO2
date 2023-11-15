#!/usr/bin/env python3
import sys
import ALICEO2dataModelTools as O2DMT

# -----------------------------------------------------------------------------
class produces:
  def __init__(self, name, templated=False):
    self.name = name
    self.templated = templated

  def tableName(self, arguments=list(), argumentValues=list()):
    tname = ""
    if len(arguments) != len(argumentValues):
      print("ATTENTION:")
      print("  ",arguments)
      print("  ",argumentValues)
      return tname

    if self.templated:
      for i in range(len(arguments)):
        if self.name == arguments[i]:
          tname = argumentValues[i]
          continue
    else:
      tname = self.name

    return tname

# -----------------------------------------------------------------------------
class struct:
  def __init__(self, name, cont, templateArguments=list()):
    self.name = name
    self.cont = cont
    self.produces = list()
    self.templated = False
    self.templateArguments = list()
    if len(templateArguments) > 0:
      self.setTemplated(templateArguments)

  # set templated
  def setTemplated(self, templateArguments):
    self.templateArguments = templateArguments
    self.templated = True

  def addProduces(self, prod):
    self.produces.append(prod)

  def getTableNames(self, argumentValues=list()):
    tableNames = list()
    for prod in self.produces:
      tableName = prod.tableName(self.templateArguments,argumentValues)
      if tableName != "":
        tableNames.append(tableName)

    return tableNames

  def print(self):
    print()
    print("Struct: ", self.name)
    print("  templated: ", self.templated)
    if self.templated:
      print("    template arguments:", len(self.templateArguments))
      for arg in self.templateArguments:
        print("      ", arg)
    print("  produces: ", len(self.produces))
    for prod in self.produces:
      print("    ", prod.name)

# -----------------------------------------------------------------------------
class codeFile:
  def __init__(self, cfile):
    self.structs = list()
    self.tableNames = list()

    with open(cfile, 'r') as file:
      # read the file
      lines_in_file = file.readlines()
      content = O2DMT.pickContent(lines_in_file)
      self.tableNames = self.parseContent(content)

  def addStruct(self, struct):
    self.structs.append(struct)

# .............................................................................
# parse the content of exeCode files

  def parseContent(self, content):
    words = content[0]
    lines = content[1]

    # find template <...>
    # restLine is everything else than a templated block
    restLine = ""
    fullLine = O2DMT.block(words)
    nchFullLine = len(fullLine)
    sob = 0

    # loop over templates
    inds = [i for i, x in enumerate(words) if x.txt == 'template']
    for ind in inds:
      line = O2DMT.block(words[ind:])

      # templates can come without {} block!
      # find out what comes first '{' or ';'
      if ';' in line:
        ci = line.index(';')
        o1 = ci+1
        if '{' in line:
          o1 = line.index('{')
        if ci < o1:
          tempBlock = line[:ci]
        else:
          [oi, ci] = O2DMT.findInBrackets("{","}",line)
          tempBlock = line[:ci]
      else:
        [oi, ci] = O2DMT.findInBrackets("{","}",line)
        tempBlock = line[:ci]

      if len(tempBlock) == 0:
        continue

      # extract template arguments
      tempLine = O2DMT.lineInBrackets("<",">",line)[1:-1]
      argWords = tempLine.split(',')
      tempArgs = list()
      for arg in argWords:
        kvpair = arg.split()
        if len(kvpair) == 2:
          tempArgs.append(kvpair[-1])

      # find struct within tempBlock
      tempWords = O2DMT.split(tempBlock)
      istructs = [i for i, x in enumerate(tempWords) if x == 'struct']
      for istruct in istructs:
        # extract structBlock and find Produces within
        structBlock = O2DMT.block(tempWords[istruct:])
        newStruct = struct(tempWords[istruct+1],
          O2DMT.lineInBrackets("{","}",structBlock), tempArgs)

        structWords = O2DMT.split(newStruct.cont)
        iprods = [i for i, x in enumerate(structWords) if x == 'Produces']
        for iprod in iprods:
          # get the table name
          tname = O2DMT.lineInBrackets("<",">",O2DMT.block(structWords[iprod:],False))[1:-1]
          newProd = produces(tname, True)
          newStruct.addProduces(newProd)

        if len(iprods) > 0:
          self.addStruct(newStruct)

      # update restLine
      eob = nchFullLine - len(line)
      restLine += fullLine[sob:eob]
      sob = eob+ci+1

    # update restLine
    restLine += fullLine[sob:]

    # find struct outside of template - in restLine
    restWords = O2DMT.split(restLine)
    istructs = [i for i, x in enumerate(restWords) if x == 'struct']
    for istruct in istructs:
      # extract structBlock and find Produces within
      structBlock = O2DMT.block(restWords[istruct:])
      newStruct = struct(restWords[istruct+1],
        O2DMT.lineInBrackets("{","}",structBlock))

      structWords = O2DMT.split(newStruct.cont)
      iprods = [i for i, x in enumerate(structWords) if x == 'Produces']
      for iprod in iprods:
        # get the table name
        tname = O2DMT.lineInBrackets("<",">",O2DMT.block(structWords[iprod:],False))[1:-1]
        newProd = produces(tname, False)
        newStruct.addProduces(newProd)

      if len(iprods) > 0:
        self.addStruct(newStruct)

    # loop over structs
    tableNames = list()
    for strct in self.structs:

      # for all templated structs: find flavoured calls of structs
      if strct.templated:

        # extract flavouredStruct
        inds = [i for i, x in enumerate(restWords) if x == strct.name]
        for ind in inds:
          if "<" in restWords[ind:ind+2]:
            # extract argument values
            argValues = O2DMT.getArgumentValues(restWords[ind:])
            tableNames += strct.getTableNames(argValues)

      else:
        tableNames += strct.getTableNames()

    # uniqify tableNames
    tableNames = list(set(tableNames))

    return tableNames

# -----------------------------------------------------------------------------
