#!/usr/bin/python3.6

import os
import sys
import ALICEO2dataModel as DM
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------
# main
#
# .............................................................................
def main(initCard):

  # which action
  todo = initCard.find('action')
  if todo == None:
    todo = 1
  else:
    todo = int(todo.text)

  # O2dir and main header file
  O2dir = initCard.find('O2general/mainDir/local')
  if O2dir == None:
    return
  O2dir = O2dir.text.strip()

  DMH = initCard.find('O2general/DataModelHeader')
  if DMH == None:
    return

  fileName = O2dir+"/"+DMH.text.strip()
  mainProducer = initCard.find('O2general/producer')
  if mainProducer == None:
    mainProducer = "AO2D files"
  else:
    mainProducer = mainProducer.text.strip()

  # =============================================== main header file ============
  # read the file and create AO2D datamodel
  if todo == 1:
    print("Main header file: ", fileName)
  dm = DM.datamodel(mainProducer, ["", "", mainProducer], fileName, initCard)

  # now get additional header files with table/column declarations
  # the directories to consider
  if todo == 1:
    print()
    print("Other header files")

  # =============================================== other header files ==========
  hfMainDir = initCard.find('headerFiles/mainDir')
  if hfMainDir == None:
    hfMainDir = ""
  else:
    hfMainDir = hfMainDir.text.strip()
  hfMainDir = O2dir+"/"+hfMainDir

  hfSubDirs = initCard.find('headerFiles/subDirs')
  if hfSubDirs == None:
    hfSubDirs = ['']
  else:
    hfSubDirs = hfSubDirs.text.strip().split(",")

  hftmp = initCard.find('headerFiles/fileName')
  if hftmp == None:
    hftmp = "*.h"
  else:
    hftmp = hftmp.text.strip()

  inclfiles = list()
  for hfSubDir in hfSubDirs:
    sname = hfMainDir+"/"+hfSubDir.strip()+"/"+hftmp
    stream = os.popen("ls -1 "+sname)
    inclfiles.extend(stream.readlines())

  # loop over these header files and join the related datamodels
  # with the AO2D datamodel
  for infile in inclfiles:
    # extract datamodel name
    path = infile.split('/')[:-1]
    cfile = infile.split('/')[-1]
    CErelation = [path, cfile, ""]
    if todo == 1:
      print("  ", infile.rstrip())
    dmnew = DM.datamodel(cfile.split(".")[0], CErelation, infile.rstrip())
    dm.join(dmnew)

  # =============================================== CMakeLists.txt ==============
  # analyze CMakeLists.txt and extract code - executable relations defined
  # with o2_add_dpl_workflow
  # the directories to consider
  if todo == 1:
    print()
    print("CMakeLists.txt")

  cmMainDir = initCard.find('CMLfiles/mainDir')
  if cmMainDir == None:
    cmMainDir = ""
  else:
    cmMainDir = cmMainDir.text.strip()
  cmMainDir = O2dir+"/"+cmMainDir

  cmSubDirs = initCard.find('CMLfiles/subDirs')
  if cmSubDirs == None:
    cmSubDirs = [""]
  else:
    cmSubDirs = cmSubDirs.text.strip().split(",")

  cmtmp = initCard.find('CMLfiles/fileName')
  if cmtmp == None:
    cmtmp = "*"
  else:
    cmtmp = cmtmp.text.strip()

  cmakefiles = list()
  for cmSubDir in cmSubDirs:
    sname = cmMainDir+"/"+cmSubDir.strip()+"/"+cmtmp
    if todo == 1:
      print("  ", sname)
    stream = os.popen("ls -1 "+sname)
    cmakefiles.extend(stream.readlines())

  cerelations = DM.CERelations(initCard)
  for cfile in cmakefiles:
    cfile = cfile.rstrip("\n")
    cerelations.addRelations(cfile)

  # =============================================== code files ==================
  # get a list of producer code files (*.cxx)
  # the directories to consider
  if todo == 1:
    print()
    print("Code files")

  codeMainDir = initCard.find('codeFiles/mainDir')
  if codeMainDir == None:
    codeMainDir = ""
  else:
    codeMainDir = codeMainDir.text.strip()
  codeMainDir = O2dir+"/"+codeMainDir

  codeSubDirs = initCard.find('codeFiles/subDirs')
  if codeSubDirs == None:
    codeSubDirs = [""]
  else:
    codeSubDirs = codeSubDirs.text.strip().split(",")

  codetmp = initCard.find('codeFiles/fileName')
  if codetmp == None:
    codetmp = "*"
  else:
    codetmp = codetmp.text.strip()

  codefiles = list()
  for codeSubDir in codeSubDirs:
    sname = codeMainDir+"/"+codeSubDir.strip()+"/"+codetmp
    stream = os.popen("grep -l Produces "+sname)
    cfiles = stream.readlines()
    codefiles.extend(cfiles)
    if todo == 1:
      for cfile in cfiles:
        print("  ", cfile.rstrip("\n"))

  # loop over these code files an find out which tables they produce
  # update the data model accordingly using setProducer
  for codefile in codefiles:
    codefile = codefile.rstrip("\n")
    CErelation = cerelations.getExecutable(codefile)
    stream = os.popen("grep Produces "+codefile)
    prods = stream.readlines()
    for prod in prods:
      prod = prod.rstrip("\n").strip()
      tableName = DM.fullDataModelName("o2::aod", prod.split("<")[-1].split(">")[0])
      dm.setProducer(CErelation, tableName)

  # =============================================== print out ===================
  # print the data model
  if todo == 2:
    dm.print()
  if todo == 3:
    dm.printHTML()


# -----------------------------------------------------------------------------
if __name__ == "__main__":

  initCard = ET.parse("inputCard.xml")

  main(initCard)

# -----------------------------------------------------------------------------
