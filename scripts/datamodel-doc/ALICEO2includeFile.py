#!/usr/bin/env python3
import sys
import os
import numpy as np
import re
import ALICEO2dataModelTools as O2DMT

# -----------------------------------------------------------------------------
# definitions
#
# .............................................................................
# types of column declarations
#  0: COLUMN
#  1: INDEX_COLUMN_FULL
#  2: INDEX_COLUMN
#  3: SELF_INDEX_COLUMN_FULL
#  4: SELF_INDEX_COLUMN
#  5: EXPRESSION_COLUMN
#  6: DYNAMIC_COLUMN
#  7: SLICE_INDEX_COLUMN
#  8: SLICE_INDEX_COLUMN_FULL
#  9: SELF_SLICE_INDEX_COLUMN
# 10: SELF_ARRAY_INDEX_COLUMN

def columnTypes(abbr=0):
  if abbr == 0:
    types = ["", "INDEX_", "INDEX_", "INDEX_", "INDEX_", "EXPRESSION_", "DYNAMIC_", "SLICE_INDEX_", "SLICE_INDEX_", "SLICE_INDEX_", "ARRAY_INDEX_"]
    types[3] = "SELF_"+types[3]
    types[4] = "SELF_"+types[4]
    types[9] = "SELF_"+types[9]
    types[10] = "SELF_"+types[10]
    types = [s+"COLUMN" for s in types]
    types = ["DECLARE_SOA_"+s for s in types]
    types[1] = types[1]+"_FULL"
    types[3] = types[3]+"_FULL"
    types[8] = types[8]+"_FULL"
  else:
    # always add "GI" as last element
    types = ["", "I", "I", "SI", "SI", "E", "D", "SLI", "SLI", "SSLI", "SAI", "GI"]

  return types

# .............................................................................
# types of table declarations
# 0: TABLE
# 1: TABLE_VERSIONED
# 2: TABLE_FULL
# 3: TABLE_FULL_VERSIONED
# 4: EXTENDED_TABLE
# 5: INDEX_TABLE
# 6: INDEX_TABLE_EXCLUSIVE
# 7: EXTENDED_TABLE_USER

def tableTypes(abbr=0):
  if abbr == 0:
    types = ["", "", "", "", "EXTENDED_", "INDEX_", "INDEX_", "EXTENDED_"]
    types = [s+"TABLE" for s in types]
    types = ["DECLARE_SOA_"+s for s in types]
    types[1] = types[1]+"_VERSIONED"
    types[2] = types[2]+"_FULL"
    types[3] = types[3]+"_FULL_VERSIONED"
    types[6] = types[6]+"_EXCLUSIVE"
    types[7] = types[7]+"_USER"
  else:
    types = ["", "", "", "", "E", "I", "I", "E"]

  return types

# -----------------------------------------------------------------------------
# classes
#
# .............................................................................
# holds a typedef

class typedef:
  def __init__(self, name, cont):
    self.name = name
    self.cont = cont

  def print(self):
    print("    typedef: "+self.name)
    print("      content: "+self.cont)

# .............................................................................
# holds a using

class using:
  def __init__(self, nslevel, name, definition, cont):
    self.nslevel = nslevel
    self.name = name
    self.definition = definition
    self.cont = cont
    self.kind = 0
    self.master = ""
    self.joiners = list()

    self.master = fullDataModelName(nslevel, definition)
    if definition.find("::iterator") != -1:
      self.kind += 1
      self.master = fullDataModelName(
          nslevel, "::".join(definition.split("::")[:-1]))
    if definition.find("::Join<") != -1:
      self.kind += 2
      tmp = definition.split("<")[1]
      tmp = tmp.split(">")[0]
      self.joiners = [fullDataModelName(
          self.nslevel, s) for s in tmp.split(",")]

  # is this included
  def where(self, usings):
    for ind in range(len(usings)):
      if self.name == usings[ind].name:
        return ind
    return -1

  def synchronize(self, dm):
    # check if any master equals a using name
    if self.kind & 1 > 0:
      for nsp in dm.namespaces:
        for use in nsp.usings:
          if self.master == use.name:
            if len(use.joiners) > 0:
              self.kind += 2
              self.joiners = use.joiners

  def print(self):
    print("    using: "+self.name)
    print("      kind: ", self.kind)
    print("      definition: ", self.definition)
    if self.kind & 1 > 0:
      print("      master: "+self.master)
    if self.kind & 2 > 0:
      print("      joiners: ", self.joiners)

  def printHTML(self):
    toPrint = self.name+" = "
    if self.kind & 2 > 0:
      toPrint += "soa::Join<"
      for ind in range(len(self.joiners)):
        if ind == 0:
          toPrint += self.joiners[ind]
        else:
          toPrint += ", "+self.joiners[ind]
      toPrint += ">"
    else:
      toPrint += self.master

    if self.kind & 1 > 0:
      toPrint += "::iterator"

    print("        <li>"+toPrint+"</li>")

# -----------------------------------------------------------------------------
# holds a column

class column:
  def __init__(self, kind, nslevel, hfile, cname, gname, type, cont):
    self.kind = kind
    self.nslevel = nslevel
    self.hfile = hfile        # header file
    self.cname = cname        # column name
    self.gname = gname        # getter name
    self.type = type
    self.cont = cont
    self.pointsInto = ""
    self.comment = ""

  def where(self, columns):
    for ind in range(len(columns)):
      if self.cname == columns[ind].cname:
        return ind
    return -1

  def print(self):
    print("         ns: "+self.nslevel)
    print("     column: "+self.cname)
    print("       kind: ", self.kind)
    print("     access: "+self.gname)
    print("       type: "+self.type)
    print("header file: "+self.hfile)
    print("    comment: "+self.comment)

  def printHTML(self):
    cn2u = fullDataModelName(self.nslevel, self.cname)

    # some columns don't need to be printed
    cols2Skip = [ "o2::soa::Marker" ]
    if not any(cn2u.startswith(word) for word in cols2Skip):
      cn2u = cn2u.replace(":collision",":&zwnj;collision")
      # replace < by &lt; and > by &gt;
      ty2u = self.type.replace("<","&lt;").replace(">","&gt;")
      print("      <tr>")
      print("        <td>"+cn2u+"</td>")
      print("        <td>"+columnTypes(1)[self.kind]+"</td>")
      print("        <td>"+self.gname+"</td>")
      print("        <td>"+ty2u+"</td>")
      print("        <td>"+self.comment+"</td>")
      print("      </tr>")


# .............................................................................
# holds a table

class table:
  def __init__(self, kind, nslevel, hfile, tname, cont):
    self.kind = kind
    self.nslevel = nslevel
    self.hfile = hfile          # header file
    self.tname = tname          # table name
    self.CErelations = list()
    self.cont = cont
    self.colNames = list()
    self.columns = list()
    self.toExtendWith = ""
    self.comment = ""

  def addColumn(self, col):
    self.columns.append(col)

  def where(self, tables):
    for ind in range(len(tables)):
      if self.tname == tables[ind].tname:
        return ind
    return -1

  # fill columns
  def synchronize(self, dm):
    for colName in self.colNames:
      # get column with name colName
      col = dm.getColumn(colName)
      if col not in self.columns:
        self.columns.append(col)

  def print(self):
    print("    table: "+self.tname)
    print("   header file: ", self.hfile)
    print("          kind: ", self.kind)
    print("     producers: ", len(self.CErelations))
    for cer in self.CErelations:
      print("             ", cer[2])
    for col in self.columns:
      print("        column: "+col.cname+":"+col.type)

  def printHeaderHTML(self):
    tableName = self.tname
    if tableTypes(1)[self.kind] != "":
      tableName += " ("+tableTypes(1)[self.kind]+")"
    print("  <button class=\"myaccordion\"><i class=\"fa fa-table\"></i> " +
          tableName+"</button>")
    print("  <div class=\"panel\">")

  def printSubHeaderHTML(self):
    print("    <table class=DataModel>")
    print("      <tr>")
    print("        <th>Name</th>")
    print("        <th></th>")
    print("        <th>Getter</th>")
    print("        <th>Type</th>")
    print("        <th>Comment</th>")
    print("      </tr>")

  def printFooterHTML(self):
    print("    </table>")
    print("  </div>")
    print("")


# -----------------------------------------------------------------------------
# holds a namespace

class namespace:
  def __init__(self, nslevel, cont):
    self.nslevel = nslevel
    self.cont = cont
    self.columns = list()
    self.tables = list()
    self.usings = list()

  def addUsing(self, using):
    self.usings.append(using)

  def addColumn(self, col):
    self.columns.append(col)

  def addTable(self, table):
    self.tables.append(table)

  # set the producer
  def setProducer(self, CErelation, tableName):
    for ind in range(len(self.tables)):
      if self.tables[ind].tname == tableName:
        if not CErelation in self.tables[ind].CErelations:
          self.tables[ind].CErelations.append(CErelation)

  # fill columns of all tables
  def synchronize(self, dm):
    for use in self.usings:
      use.synchronize(dm)
    for tab in self.tables:
      tab.synchronize(dm)

  # merge with newnsp
  def join(self, newnsp):
    if newnsp.nslevel != self.nslevel:
      return
    for use in newnsp.usings:
      ind = use.where(self.usings)
      if ind < 0:
        self.usings.append(use)
    for tab in newnsp.tables:
      ind = tab.where(self.tables)
      if ind < 0:
        self.tables.append(tab)
    for col in newnsp.columns:
      ind = col.where(self.columns)
      if ind < 0:
        self.columns.append(col)

  def print(self):
    print("  namespace: "+self.nslevel)
    print("    number of tables: ", len(self.tables))
    for tab in self.tables:
      tab.print()
    print("    number of usings: ", len(self.usings))
    for use in self.usings:
      use.print()
    print("    number of columns: ", len(self.columns))
    for col in self.columns:
      col.print()

# -----------------------------------------------------------------------------
# holds a datamodel

class datamodel:
  def __init__(self, dmname, CErelation, hfile, initCard=None):
    with open(hfile, 'r') as file:
      self.dmname = dmname
      self.CErelations = list()
      self.CErelations.append(CErelation)
      self.defines = list()
      self.namespaces = list()
      self.categories = list()

      # set some variables
      self.O2path = ""
      self.O2Physicspath = ""
      self.O2href = ""
      self.O2Physicshref = ""
      self.delimAO2D = ""
      self.delimHelpers = ""
      self.delimPWGs = ""
      self.delimJoins = ""
      # update with values from initCard
      if initCard != None:
        psep = os.path.sep
        self.initCard = initCard
        tmp = initCard.find("O2general/mainDir/O2local")
        if tmp != None:
          self.O2path = tmp.text.strip().rstrip(psep)+psep
        tmp = initCard.find("O2general/mainDir/O2Physicslocal")
        if tmp != None:
          self.O2Physicspath = tmp.text.strip().rstrip(psep)+psep
        tmp = initCard.find("O2general/mainDir/O2GitHub")
        if tmp != None:
          self.O2href = tmp.text.strip().rstrip(psep)+psep
        tmp = initCard.find("O2general/mainDir/O2PhysicsGitHub")
        if tmp != None:
          self.O2Physicshref = tmp.text.strip().rstrip(psep)+psep
        tmp = initCard.find("O2general/delimAO2D")
        if tmp != None:
          self.delimAO2D = tmp.text.strip()
        tmp = initCard.find("O2general/delimHelpers")
        if tmp != None:
          self.delimHelpers = tmp.text.strip()
        tmp = initCard.find("O2general/delimPWGs")
        if tmp != None:
          self.delimPWGs = tmp.text.strip()
        tmp = initCard.find("O2general/delimJoins")
        if tmp != None:
          self.delimJoins = tmp.text.strip()

      # read the file
      lines_in_file = file.readlines()
      content = O2DMT.pickContent(lines_in_file)

      # parse datamodel
      self.parseContent(hfile, content, "", self)
      # self.synchronize()

  # extract the categories definition
  def setTableCategories(self, DMxml):

    # table categories
    cats = DMxml.find('categories')
    for cat in cats:
      catName = cat.attrib['name']
      catTables = "".join(cat.text.split()).split(",")
      self.categories.append(tableCategory(catName,catTables))

  # A namespace is contained between "namespace 'name' {" and "}"
  # Be aware that namespaces can be nested!
  def parseContent(self, hfile, content, nslevel, dm):
    words = content[0]
    lines = content[1]

    # does this block contain a namespace definition?
    # 2 formats
    #   1. using namespace .....;
    #   2. namespace .... {}
    isps = [ind for ind, x in enumerate(words) if x.txt == "namespace"]
    if len(isps) > 0:
      p10 = isps[0]
      if words[p10-1].txt == "using":
        # 1. using namespace .....;
        iop = [ind for ind, x in enumerate(words[p10:]) if x.txt == ";"]
        if len(iop) == 0:
          print("using namespace does not end with \";\"!")
          print(O2DMT.block(words[p10:]))
          exit()
        p11 = len(words)

      else:
        # 2. namespace .... {}
        iop = [ind for ind, x in enumerate(words[p10:]) if x.txt == "{"]
        if len(iop) == 0:
          print("The opening bracket \"{\" is missing!")
          print(O2DMT.block(words[p10:]))
          exit()
        icl = [ind for ind, x in enumerate(words[p10:]) if x.txt == "}"]
        if len(icl) == 0:
          print("The closing bracket \"}\" is missing!")
          print(O2DMT.block(words[p10:]))
          exit()

        # find namespace block within {}
        nind = len(words) - p10
        ind = np.zeros(nind)
        ind[iop] = 1
        ind[icl] = -1
        p11 = np.where(np.cumsum(ind[iop[0]:]) == 0)
        if len(p11[0]) <= 0:
          print(hfile)
          exit()
        p11 = p10+iop[0]+p11[0][0]

      # analyze the next block with updated nslevel
      b2u = O2DMT.block(words[p10+1:p10+iop[0]], False)
      if nslevel != "":
        nslnew = fullDataModelName(nslevel, b2u)
      else:
        nslnew = b2u

      c2u = O2DMT.select(content, p10+1, p11)
      self.parseContent(hfile, c2u, nslnew, dm)

      # remove words of ns and process rest
      if p10 > 0 and p11 < len(words):
        c2u = O2DMT.select(content, 0, p10, p11+1)
        self.parseContent(hfile, c2u, nslevel, dm)

    else:
      # this block of text is a namespace
      if nslevel == "":
        return True

      # find typedefs and replace affected items
      itds = [ind for ind, x in enumerate(words) if x.txt == "typedef"]
      for itd in itds:
        name1 = words[itd+1].txt
        name2 = words[itd+2].txt
        # replace all name2 with name1
        for ind in range(itd+3, len(words)):
          if words[ind].txt == name2:
            words[ind].txt = name1

      # analyze the block and create a namespace object nsp
      nsp = namespace(nslevel, O2DMT.block(words))

      # extract columns
      cols = extractColumns(nslevel, content)
      for col in cols:
        col.hfile = hfile
        nsp.addColumn(col)

      # extract tables
      tables = extractTables(nslevel, content)
      for tab in tables:
        tab.CErelations.append(dm.CErelations[0])
        tab.hfile = hfile
        nsp.addTable(tab)

      # extract usings
      usings = extractUsings(nslevel, content)
      for using in usings:
        nsp.addUsing(using)

      # update the datamodel dm
      dm.addNamespace(nsp)

    return True

  def addNamespace(self, namespace):
    # does this namespace exist already?
    ind = self.where(namespace)
    if ind >= 0:
      self.namespaces[ind].join(namespace)
    else:
      self.namespaces.append(namespace)

  def addDefine(self, define):
    self.defines.append(define)

  def setProducer(self, CErelation, tableName):
    for nsp in self.namespaces:
      nsp.setProducer(CErelation, tableName)
      if not CErelation in self.CErelations:
        self.CErelations.append(CErelation)

  def isProducedBy(self, CErelation):
    producedBy = list()
    for nsp in self.namespaces:
      for table in nsp.tables:
        if CErelation in table.CErelations:
          producedBy.append(table)
    return producedBy

  def where(self, namespace):
    for ind in range(len(self.namespaces)):
      if namespace.nslevel == self.namespaces[ind].nslevel:
        return ind
    return -1

  # join with data model dmnew
  def join(self, dmnew):
    for nsp in dmnew.namespaces:
      ind = self.where(nsp)
      if ind >= 0:
        self.namespaces[ind].join(nsp)
      else:
        self.namespaces.append(nsp)

  # fill columns of all tables
  def synchronize(self):
    for nsp in self.namespaces:
      nsp.synchronize(self)

  def getColumn(self, colName):
    # remove <> from colName
    cnameToSearch = colName.split("<")[0]

    # index columns are special!
    if cnameToSearch == "o2::soa::Index":
      return column(len(columnTypes(1))-1, "o2::soa", "", cnameToSearch, "globalIndex", "int64_t", colName)

    for nsp in self.namespaces:
      for col in nsp.columns:
        cnameToUse = fullDataModelName(nsp.nslevel, col.cname)
        if cnameToUse == cnameToSearch:
          return col

    # create dummy column
    return column(-1, "", "", cnameToSearch, "", "?", colName)

  def print(self):
    print("data model: "+self.dmname)
    print("  producers:")
    for CErelation in self.CErelations:
      print("    ", CErelation[2])
    for ns in self.namespaces:
      ns.print()

  def printSingleTable(self, tabs, uses, tab2u):
     # print the table header
    tab2u.printHeaderHTML()

    # print table comment
    print("    <div>")
    print("      ", tab2u.comment)
    print("    </div>")

    # print header file
    if "O2Physics" in tab2u.hfile:
      href2u = self.O2Physicshref
      path2u = self.O2Physicspath
    else:
      href2u = self.O2href
      path2u = self.O2path

    hf2u = O2DMT.block(tab2u.hfile.split(path2u)[
                 1:], False).strip().lstrip("/")
    print("    <div>")
    print("      Header file: <a href=\""+href2u +
          "/"+hf2u+"\" target=\"_blank\">"+hf2u+"</a>")
    print("    </div>")

    # print extends
    if tab2u.kind == 4 or tab2u.kind == 7:
      print("    <div>Extends:")
      print("      <ul>")
      print("        ", tab2u.toExtendWith)
      print("      </ul>")
      print("    </div>")

    # find all usings with tab2u
    useTable = list()
    for use in uses:
      if tab2u.tname in use.joiners:
        useTable.append(use)
      elif tab2u.tname == use.master:
        useTable.append(use)

    # print these usings
    if len(useTable) > 0:
      print("    <div>Is used in:")
      print("      <ul>")
      for use in useTable:
        use.printHTML()
      print("      </ul>")
      print("    </div>")

    # print the table header
    tab2u.printSubHeaderHTML()

    # EXTENDED_TABLE and EXTENDED_TABLE_USER are extended
    if tab2u.kind == 4 or tab2u.kind == 7:
      # this table has to be extended, find the extending table and
      # print all of its columns
      einds = [i for i, x in enumerate(tabs) if x.tname == tab2u.toExtendWith]
      for ind in einds:
        for col in tabs[ind].columns:
          col.printHTML()

    # print the remaining columns
    for col in tab2u.columns:
      col.printHTML()

    # print the table footer
    tab2u.printFooterHTML()


  def printTables(self, DMtype, tabs, uses, CER, tabs2u):
    print("")
    print("#### ", CER[2])

    # add source code information if available
    if "O2Physics" in CER[0]:
      href2u = self.O2Physicshref
      path2u = self.O2Physicspath
    else:
      href2u = self.O2href
      path2u = self.O2path

    if DMtype == 1:
      if href2u != "":
        print("Code file: <a href=\""+href2u+"/"+CER[0].split(path2u)[1] +
              "/"+CER[1]+"\" target=\"_blank\">"+CER[1]+"</a>")
      else:
        print("Code file: "+CER[0]+"/"+CER[1])

    tabInCat = list()
    others = list()
    if DMtype == 0:
      # pattern for table versions
      vPattern = self.initCard.find('O2general/TableVersionPattern')
      if vPattern == None:
        vPattern = "_\d\d\d$"
      else:
        vPattern = vPattern.text.strip()

      # Analyze the tables and categories
      tabInCat = [False]*len(tabs2u)
      for cat in self.categories:
        for i in range(0,len(tabs2u)):
          if baseTableName(tabs2u[i].tname, vPattern) in cat.members:
            tabInCat[i] = True
      others = [i for i, x in enumerate(tabInCat) if x == False]

      # print available categories
      txt2print = "For better overview the tables are grouped into the following categories: |"
      for cat in self.categories:
        txt2print = txt2print+' ['+cat.name+'](#cat_'+cat.name+') |'
      if len(others) > 0:
        txt2print = txt2print+' [Others](#cat_Others) |'
      print(txt2print)
      print()

    print("<div>")
    print("")

    # loop over all table categories
    if DMtype == 0:
      for cat in self.categories:
        txt2print = '<h4 id="cat_'+cat.name+'">'+cat.name+'</h4>'
        print(txt2print)
        print("<div>")

        # print tables of of given category
        for tname in cat.members:
          for tab in tabs2u:
            if baseTableName(tab.tname, vPattern) == tname:
              print()
              self.printSingleTable(tabs, uses, tab)
              continue
        print("</div>")

      # print non-categorized tables
      if len(others) > 0:
        print('<h4 id="cat_Others">Others</h4>')
        print("<div>")
        for i in others:
          print()
          self.printSingleTable(tabs, uses, tabs2u[i])
        print("</div>")

    else:
      # print all tables of given producer
      for tab in tabs2u:
        self.printSingleTable(tabs, uses, tab)

    print("</div>")

  def printHTML(self):
    # gather all tables and columns
    tabs = list()
    uses = list()
    for nsp in self.namespaces:
      for tab in nsp.tables:
        tabs.append(tab)
      for use in nsp.usings:
        uses.append(use)

    # Create html documents in 4 steps
    # 1. main producer
    # 2. helper tasks
    # 3. PWG tasks
    # 4. joins

    # 1. main producer
    print(self.delimAO2D)
    inds = [i for i, x in enumerate(self.CErelations) if x[3] == 'Main']
    CER2u = [self.CErelations[i] for i in inds]
    # only one Main CER should be available
    if len(CER2u) != 1:
      sys.exit('<datamodel.printHTML> Exacly 1 DataModel of type Main is expected. We found '+len(CER2u)+'! EXIT -->')

    for CER in CER2u:
      inds = [i for i, x in enumerate(tabs) if CER in x.CErelations]
      tabs2u = [tabs[i] for i in inds]
      self.printTables(0, tabs, uses, CER, tabs2u)
    print(self.delimAO2D)

    # 2. helper tasks
    print("")
    print(self.delimHelpers)
    inds = [i for i, x in enumerate(self.CErelations) if x[3] == 'Helper']
    CER2u = [self.CErelations[i] for i in inds]
    for CER in CER2u:
      inds = [i for i, x in enumerate(tabs) if CER in x.CErelations]
      tabs2u = [tabs[i] for i in inds]
      self.printTables(1, tabs, uses, CER, tabs2u)
    print(self.delimHelpers)

    # 3. PWG tasks
    print("")
    print(self.delimPWGs)
    inds = [i for i, x in enumerate(self.CErelations) if x[3] == 'PWG']
    CERsPWG = [self.CErelations[i] for i in inds]

    # PWG data model names
    dmnames = [CERsPWG[i][4] for i in list(range(0, len(CERsPWG)))]
    dmnames = np.unique(dmnames)
    for dmname in dmnames:
      print("")
      print("##", 'PWG-'+dmname)

      inds = [i for i, x in enumerate(CERsPWG) if x[4] == dmname]
      CER2u = [CERsPWG[i] for i in inds]
      for CER in CER2u:
        inds = [i for i, x in enumerate(tabs) if CER in x.CErelations]
        tabs2u = [tabs[i] for i in inds]
        self.printTables(1, tabs, uses, CER, tabs2u)

    print(self.delimPWGs)
    print("")

    # now print the usings
    if len(uses) > 0:
      print(self.delimJoins)
      print("")
      print("<a name=\"usings\"></a>")
      print("#### List of defined joins and iterators")
      print("<div>")
      for use in uses:
        print("")
        print(
            "  <button class=\"myaccordion\"><i class=\"fa fa-map-pin\"></i> "+use.name+"</button>")
        print("  <div class=\"panel\">")
        print("    <ul>")
        use.printHTML()
        print("    </ul>")
        print("  </div>")
      print("</div>")
      print(self.delimJoins)

# -----------------------------------------------------------------------------
# functions
#
# .............................................................................
# remove the version id from the table name

def baseTableName(vtname, vPattern):

  vres = re.compile(vPattern).search(vtname)
  if vres:
    return vtname[0:vres.start()]
  else:
    return vtname

# .............................................................................

def fullDataModelName(nslevel, name):
  toks0 = nslevel.split("::")
  toks1 = name.split("::")

  fullName = ""
  for tok in toks0:
    if tok not in toks1:
      if fullName == "":
        fullName = tok
      else:
        fullName += "::"+tok
    else:
      break

  if fullName == "":
    fullName = name
  else:
    fullName += "::"+name

  return fullName

# .............................................................................
# extract the column names from a table declaration
# cont contains the declaration

def tableColumnNames(nslevel, cont, kind=0):

  # specification according to kind of table
  noffs = [3, 4, 4, 5, 3, 3, 3, 3]
  noff = noffs[kind]

  # split cont with ","
  buf = O2DMT.block(cont[:len(cont)-2], False)
  toks = buf.split(",")

  # get column definitions, ATTENTION: some contain <>
  colNames = list()
  col = ""
  nop = 0
  ncl = 0
  for tok in toks[noff:]:
    col += tok
    nop += tok.count("<")
    ncl += tok.count(">")

    if (ncl == nop):
      colNames.append(col)
      col = ""
    else:
      col += ","

  # complete namespace
  fullColNames = list()
  toks0 = nslevel.split("::")
  for colName in colNames:
    toks1 = colName.split("::")

    fullColName = ""
    for tok in toks0:
      if tok not in toks1:
        if fullColName == "":
          fullColName = tok
        else:
          fullColName += "::"+tok
      else:
        break

    if fullColName == "":
      fullColName = colName
    else:
      fullColName += "::"+colName

    fullColNames.append(fullColName)

  return fullColNames

# .............................................................................
# extract table definitions from content

def extractTables(nslevel, content):
  words = content[0]
  lines = content[1]
  tables = list()

  # table types
  types = tableTypes()

  # find indices of any type of table declarations
  def condition(s):
    res = False
    for x in types:
      res = res or (s.txt == x)
    return res
  inds = [idx for idx, element in enumerate(words) if condition(element)]

  # loop over declarations
  for icol in inds:
    iend = [i for i, x in enumerate(
        O2DMT.list_in([")", ";"], words[icol:])) if x == True]
    if len(iend) == 0:
      print(nslevel)
      sys.exit('Ending ); not found in table declaration! EXIT -->')
    cont = words[icol:iend[0]+icol+2]

    kind = [i for i, x in enumerate(types) if x == words[icol].txt][0]
    tname = fullDataModelName(nslevel, words[icol+2].txt)

    # extract column names
    fullColNames = tableColumnNames(nslevel, cont, kind)

    # kind, namespace, tname, cont
    tab = table(kind, nslevel, "", tname, O2DMT.block(cont))
    tab.colNames = fullColNames

    # EXTENDED_TABLE?
    if kind == 4 or kind == 7:
      tab.toExtendWith = fullDataModelName(nslevel, words[icol+4].txt)

    # add a comment if available
    line = lines[words[icol].lnr]
    tab.comment = O2DMT.block(line.split("//!")[1:], True).strip()

    tables.append(tab)

  return tables

# .............................................................................
# extract the column definitions from content

def extractColumns(nslevel, content):
  words = content[0]
  lines = content[1]

  # helper function to find any table declaration in a list of words
  def condition(s):
    res = False
    for x in types:
      res = res or (s.txt == x)
    return res

  cols = list()

  # column types
  types = columnTypes()

  # find indices of column declarations
  inds = [idx for idx, element in enumerate(words) if condition(element)]

  # loop over declarations
  for icol in inds:
    iend = [i for i, x in enumerate(
        O2DMT.list_in([")", ";"], words[icol:])) if x == True]
    if len(iend) == 0:
      print(nslevel)
      sys.exit('Ending ); not found in column declaration! EXIT -->')
    cont = words[icol:iend[0]+icol+2]

    kind = [i for i, x in enumerate(types) if x == words[icol].txt][0]
    cname = words[icol+2].txt
    gname = words[icol+4].txt
    if kind in [1, 2, 3, 4]:
      cname = cname+"Id"
      gname = gname+"Id"
    if kind in [7,8,9]:
      cname = cname+"IdSlice"
      gname = gname+"Ids"
    if kind in [10]:
      cname = cname+"Ids"
      gname = gname+"Ids"

    # determine the type of the colums
    # can be type, array<type,n>, or type[n]
    type = ""
    if words[icol].txt == types[0]:
      type = O2DMT.block(words[icol+6:icol+iend[0]], False)
    elif words[icol].txt == types[1]:
      type = words[icol+6].txt
    elif words[icol].txt == types[2]:
      type = "int32"
    elif words[icol].txt == types[3]:
      type = words[icol+6].txt
    elif words[icol].txt == types[4]:
      type = words[icol+6].txt
    elif words[icol].txt == types[5]:
      iend = [i for i, x in enumerate(
          O2DMT.list_in([","], words[icol+6:])) if x == True]
      type = O2DMT.block(words[icol+6:icol++6+iend[0]], False)
    elif words[icol].txt == types[6]:
      iarr = [i for i, x in enumerate(
          O2DMT.list_in(["-", ">"], cont)) if x == True]
      if len(iarr) > 0:
        iend = [i for i, x in enumerate(
            O2DMT.list_in(["{"], cont[iarr[0]+2:])) if x == True]
        type = O2DMT.block(cont[iarr[0]+2:iarr[0]+2+iend[0]], False)
      else:
        type = "?"
    elif words[icol].txt in types[7:10]:
      type = "int32_t"

    # kind, namespace, name, type, cont
    col = column(kind, nslevel, "", cname, gname, type, O2DMT.block(cont))
    if kind == 1:
      col.pointsInto = words[icol+8].txt
    if kind == 2:
      col.pointsInto = words[icol+2].txt+"s"

    # add a comment if available
    comment = ""
    if kind in [1, 2]:
      comment = "Pointer into "+col.pointsInto
    line = lines[words[icol].lnr]
    toks = line.split("//!")
    if len(toks) > 1:
      tmp = O2DMT.block(toks[1:], True).strip()
      if tmp != "":
        comment = tmp
    col.comment = comment

    cols.append(col)

  return cols

# .............................................................................
# extracts the using definitions from content

def extractUsings(nslevel, content):
  words = content[0]
  lines = content[1]
  usings = list()

  # using types
  types = ["using"]

  def condition(s):
    res = False
    for x in types:
      res = res or (s.txt == x)
    return res

  # find indices of "using"
  inds = [idx for idx, element in enumerate(words) if condition(element)]

  # loop over cases
  for icol in inds:
    iend = [i for i, x in enumerate(
        O2DMT.list_in([";"], words[icol:])) if x == True]
    if len(iend) == 0:
      print(nslevel)
      sys.exit('Ending ; not found in using declaration! EXIT -->')
    cont = words[icol:icol+iend[0]+1]

    name = fullDataModelName(nslevel, words[icol+1].txt)
    definition = O2DMT.block(words[icol+3:icol+iend[0]], False)

    # namespace, name, cont
    use = using(nslevel, name, definition, O2DMT.block(cont))

    usings.append(use)

  return usings

# -----------------------------------------------------------------------------
# A CErelation is a tuple<string,3>
#   [0]: path
#   [1]: code file (without path)
#   [2]: executable
#   [3]: type: Main, Helper, PWG

class CERelations:
  def __init__(self, initCard):
    self.relations = list()

    # exePreamble from initCard
    self.exePreamble = initCard.find('O2general/exePreamble')
    if self.exePreamble == None:
      self.exePreamble = ""
    else:
      self.exePreamble = self.exePreamble.text.strip()

    # CEdeclarationString from initCard
    self.CEdeclarationString = initCard.find(
        'O2general/CEdeclarationString')
    if self.CEdeclarationString == None:
      self.CEdeclarationString = "o2physics_add_dpl_workflow"
    else:
      self.CEdeclarationString = self.CEdeclarationString.text.strip()

  def addRelations(self, fileName, ptype, dmname):
    path = O2DMT.block(fileName.split("/")[:-1], True, "/")
    with open(fileName, 'r') as file:
      # read the file
      lines_in_file = file.readlines()
      # skip commented lines (starting with #)
      lines_in_file = [i for i in lines_in_file if not i.startswith("#")]
      # extract content
      content = O2DMT.pickContent(lines_in_file)

      # parse CMakeLists file
      # executable - code relations are defined with o2_add_dpl_workflow
      idef = [ind for ind, x in enumerate(
          content[0]) if x.txt == self.CEdeclarationString]
      for ind in idef:
        # PWG needs extra treatment
        if ptype == "PWG":
          ename = self.exePreamble + dmname.lower() + "-" + content[0][ind+2].txt
        else:
          ename = self.exePreamble + content[0][ind+2].txt
        cname = content[0][ind+4].txt
        if len(cname.split(".")) < 2:
          cname += ".cxx"
        self.relations.append([path, cname, ename, ptype, dmname])

  def getExecutable(self, codeFile):
    # find the executable corresponding to codeFile
    CErelation = ["", "", ""]
    ice = [ind for ind, x in enumerate(
        self.relations) if x[0]+x[1] == codeFile]
    if len(ice) > 0:
      CErelation = self.relations[ice[0]]
    return CErelation

  def getCodeFile(self, executable):
    # find the code file corresponding to executable
    CErelation = ["", "", ""]
    ice = [ind for ind, x in enumerate(
        self.relations) if x[2] == executable]
    if len(ice) > 0:
      CErelation = self.relations[ice[0]]
    return CErelation

  def print(self):
    print("CE relations")
    for relation in self.relations:
      print(" path  :", relation[0])
      print("  cname:", relation[1])
      print("  ename:", self.exePreamble+relation[2])
      print("   type:", relation[3])
      print("   name:", relation[4])

# -----------------------------------------------------------------------------
class tableCategory:
  def __init__(self, catName, catMembers):
    self.name = catName
    self.members = catMembers

  def blongsTo(self, tableName):
    if tableName in catMembers:
      return true
    else:
      return false

# -----------------------------------------------------------------------------
