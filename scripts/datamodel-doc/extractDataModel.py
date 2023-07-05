#!/usr/bin/env python3

import os
import sys
import ALICEO2includeFile as O2IF
import ALICEO2codeFile as O2CF
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------
# mainDataModel
#
# .............................................................................
def mainDataModel(DMs, initCard, todo=0):
  # O2dir and main header file
  O2dir = initCard.find('O2general/mainDir/O2local')
  if O2dir == None:
    return None
  O2dir = O2dir.text.strip()

  mainProducer = initCard.find('O2general/producer')
  if mainProducer == None:
    mainProducer = "AO2D files"
  else:
    mainProducer = mainProducer.text.strip()

  # find DataModel of type='Main'
  dm = None
  for subDM in DMs:
    ptype = subDM.attrib['type']
    if ptype == 'Main':
      dmname = subDM.attrib['name']
      fileName = subDM.find('headerFiles/fileName')
      if fileName != None:
        fileName = O2dir+'/'+fileName.text.strip()
        if todo == 1:
          print("  name: ", fileName)
        dm = O2IF.datamodel(mainProducer, ["", "", mainProducer, ptype, dmname], fileName, initCard)
        dm.setTableCategories(subDM)
        break

  return dm

# -----------------------------------------------------------------------------
# updateDataModel
#
# .............................................................................
def updateDataModel(O2Physicsdir, dm, subDM, todo=0):
  ptype = subDM.attrib['type']
  dmname = subDM.attrib['name']

  hfMainDir = subDM.find('headerFiles/mainDir')
  if hfMainDir == None:
    hfMainDir = ""
  else:
    hfMainDir = hfMainDir.text.strip()
  hfMainDir = O2Physicsdir+"/"+hfMainDir

  hftmps = subDM.find('headerFiles/fileName')
  if hftmps == None:
    hftmps = "*.h"
  else:
    hftmps = hftmps.text.strip()

  inclfiles = list()
  sname = ""
  for hftmp in hftmps.split(','):
    sname = sname+" "+hfMainDir+"/"+hftmp.strip()
  stream = os.popen("ls -1 2> null"+sname)
  inclfiles.extend(stream.readlines())

  # loop over these header files and join the related datamodels
  # with the dm
  for infile in inclfiles:
    if todo == 1:
      print("    ", infile.rstrip())

    # extract datamodel name
    path = infile.split('/')[:-1]
    cfile = infile.split('/')[-1]
    CErelation = [path, cfile, "", ptype, dmname]
    dmnew = O2IF.datamodel(cfile.split(".")[0], CErelation, infile.rstrip())
    dm.join(dmnew)

  return True

# -----------------------------------------------------------------------------
# addCERelations
#
# .............................................................................
def addCERelations(O2Physicsdir, cerelations, subDM, todo=0):
  ptype = subDM.attrib['type']
  dmname = subDM.attrib['name']

  cmMainDir = subDM.find('CMLfiles/mainDir')
  if cmMainDir == None:
    cmMainDir = ""
  else:
    cmMainDir = cmMainDir.text.strip()
  cmMainDir = O2Physicsdir+"/"+cmMainDir

  cmtmps = subDM.find('CMLfiles/fileName')
  if cmtmps == None:
    cmtmps = "CMakeLists.txt"
  else:
    cmtmps = cmtmps.text.strip()

  cmakefiles = list()
  sname = ""
  for cmtmp in cmtmps.split(','):
    sname = sname+" "+cmMainDir+"/"+cmtmp.strip()
  if todo == 1:
    print("    ", sname)

  stream = os.popen("ls -1 2> null "+sname)
  cmakefiles.extend(stream.readlines())

  for cfile in cmakefiles:
    cfile = cfile.rstrip("\n")
    cerelations.addRelations(cfile, ptype, dmname)

  return True

# -----------------------------------------------------------------------------
# setProducers
#
# .............................................................................
def setProducers(O2Physicsdir, cerelations, dm, subDM, todo=0):
  codeMainDir = subDM.find('codeFiles/mainDir')
  if codeMainDir == None:
    codeMainDir = ""
  else:
    codeMainDir = codeMainDir.text.strip()
  codeMainDir = O2Physicsdir+"/"+codeMainDir

  codetmps = subDM.find('codeFiles/fileName')
  if codetmps == None:
    codetmps = "*.cxx"
  else:
    codetmps = codetmps.text.strip()

  codefiles = list()
  sname = ""
  for codetmp in codetmps.split(','):
    sname = sname+" "+codeMainDir+"/"+codetmp.strip()
  stream = os.popen("grep -l Produces "+sname)
  cfiles = stream.readlines()
  codefiles.extend(cfiles)
  if todo == 1:
    for cfile in cfiles:
      print("    ", cfile.rstrip("\n"))

  # loop over these code files and find out which tables they produce
  # update the data model accordingly using setProducer
  for codefile in codefiles:
    codefile = codefile.rstrip("\n")

    CErelation = cerelations.getExecutable(codefile)
    if (len(CErelation) != 5):
      continue
    codeFile = O2CF.codeFile(codefile)
    for tableName in codeFile.tableNames:
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      # ATTENTION
      # Here it is assumed that all tables are in namespace o2::aod
      # and is explicitely set so here
      #
      tableName = "o2::aod::"+tableName.split("::")[-1]
      #
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      dm.setProducer(CErelation, tableName)

  return True

# -----------------------------------------------------------------------------
# main
#
# .............................................................................
def main(initCard, todo=0):

  # DataModel definitions
  DMs = initCard.find('DataModels')
  if DMs == None:
    return

  # =============================================== main header file ============
  if todo == 1:
    print("Main header file:")

  dm = mainDataModel(DMs, initCard, todo)
  if dm == None:
    return

  # =============================================== other header files ==========
  # now get additional header files with table/column declarations
  # the directories to consider
  # O2Physicsdir
  O2Physicsdir = initCard.find('O2general/mainDir/O2Physicslocal')
  if O2Physicsdir == None:
    return
  O2Physicsdir = O2Physicsdir.text.strip()

  if todo == 1:
    print()
    print("Other header files:")

  # join Helper data models
  if todo == 1:
    print()
    print("  Helpers:")

  isOK = False
  for subDM in DMs:
    if subDM.attrib['type'] == 'Helper':
      isOK = updateDataModel(O2Physicsdir, dm, subDM, todo)
      break
  if not isOK:
    return

  # join PWG data models
  if todo == 1:
    print()
    print("  PWGs:")

  for subDM in DMs:
    if subDM.attrib['type'] == 'PWG':
      isOK = isOK & updateDataModel(O2Physicsdir, dm, subDM, todo)
  if not isOK:
    return

  # synchronize the entire datamodel
  dm.synchronize()

  # =============================================== CMakeLists.txt ==============
  # analyze CMakeLists.txt and extract code - executable relations defined
  # with o2_add_dpl_workflow
  if todo == 1:
    print()
    print("CMakeLists:")

  cerelations = O2IF.CERelations(initCard)

  # add CERelations for Helper tasks
  if todo == 1:
    print()
    print("  Helpers:")

  isOK = False
  for subDM in DMs:
    if subDM.attrib['type'] == 'Helper':
      isOK = addCERelations(O2Physicsdir, cerelations, subDM, todo)
      break
  if not isOK:
    return

  # add CERelations for PWG tasks
  if todo == 1:
    print()
    print("  PWGs:")

  for subDM in DMs:
    if subDM.attrib['type'] == 'PWG':
      isOK = isOK & addCERelations(O2Physicsdir, cerelations, subDM, todo)
  if not isOK:
    return

  # =============================================== code files ==================
  # get a list of producer code files (*.cxx)
  if todo == 1:
    print()
    print("Code files:")

  # add Helper code files
  if todo == 1:
    print()
    print("  Helpers:")

  isOK = False
  for subDM in DMs:
    if subDM.attrib['type'] == 'Helper':
      isOK = setProducers(O2Physicsdir, cerelations, dm, subDM, todo)
      break
  if not isOK:
    return

  # add PWG code files
  if todo == 1:
    print()
    print("  PWGs:")

  for subDM in DMs:
    if subDM.attrib['type'] == 'PWG':
      isOK = isOK & setProducers(O2Physicsdir, cerelations, dm, subDM, todo)
  if not isOK:
    return

  # =============================================== print out ===================
  # print the data model
  if todo == 1:
    for rel in cerelations.relations:
      for r in rel:
        print(r)
      print("")
  if todo == 2:
    dm.print()
  if todo == 3:
    dm.printHTML()


# -----------------------------------------------------------------------------
if __name__ == "__main__":

  initCard = ET.parse("inputCard.xml")

  # which action
  todo = initCard.find('action')
  if todo == None:
    todo = 1
  else:
    todo = int(todo.text)

  main(initCard, todo)

# -----------------------------------------------------------------------------
