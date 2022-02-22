<!-- doxy
\page refscriptsDatamodel-doc ALICEO2dataModel converter
/doxy -->

# ALICEO2dataModel converter

Allows to create an html representation of the ALICE O2 data model.
The generated html code can be inserted in docs/datamodel/[ao2dTables, helperTaskTables, pwgTables, joinsAndIterators].md of https://github.com/AliceO2Group/analysis-framework/.

## Internals

The ALICE O2 Data Analysis Framework is based on a number of flat tables which contain the experimental data. The tables are declared in several header files. Some of the tables are filled automatically when an AO2D file is processed others are filled by specific tasks.

The ALICEO2dataModel converter analyses these header and task code files and extracts the table and column definitions of the tables. The information is converted into a html representation which can be included in the documentation site of the O2 Analysis Framework at https://github.com/AliceO2Group/analysis-framework/.

The converter is implemented in python and consists of the python modules extractDataModel.py, ALICEO2dataModelTools.py, ALICEO2includeFile.py, ALICEO2codeFile.py. The process is configured with inputCard.xml.


### Data Items

Three types of files are considered to contain information related to the ALICE O2 Data Analysis Model:

- header files
- code files
- CMakeLists.txt

#### Header files
Header files are combed for definitions of:

- namespace
- #define
- typedef
- using
- COLUMN declarations
- TABLE declarations

In a first step the header files are scanned and comments (//) and comment blocks (/* ... */) are removed, whereas special comments (//!) are understood to be annotations of tables and columns. Blocks of line extensions (\) are reduced to a single line. 

In the such processed text #define directives are searched for and are used to substitute related text in the block starting with the actual #define directive and ending with an #undef directive or the end of the text.

In a next processing round the text is split into namespaces, taking into account that namespaces can be nested. Namespaces are defined either with 'using namespace ...;' which is then active until the end of the text or with 'namespace {....}' which makes the namespace active only within the curled brackets. The splitting into namespaces is implemented in an iterative way.

The text contained in a singular namespace (namespace not containing any other namespace) is searched for typedef specifiers which are used to substitute related types. Then column and table declarations and finally 'using' directives within the namespace block are extracted.

As a result of this procedure one obtains a 'datamodel' object with a list of namespaces with their column and table declarations and using directives.

#### Code files
Code files are combed for definitions of:

- template
- struct
- Produces

Tables have a producer. In case of the tables filled with data from the AO2D files the producer is defined to be AO2D. For tables which are filled by analysis tasks the producer is deduced from the related Produces function declaration. The code files are scanned for Produces declarations and the name of the code file is added to the respective table in the previously described 'datamodel' object. It is taken into account, that Produces can be within templated blocks.

#### CMakeLists.txt files
CMakeLists.txt files are combed for definitions of:

- o2_add_dpl_workflow (executables and the related code files)

The CMakeLists files are finally needed to relate a given code file with an executable, which allows to associate a table with its producing executable.

### Output

The html representation of the 'datamodel' object is achieved with the printHTML method.


## As simple as that

- Install the software

Clone O2 as usual. Go to scripts/datamodel-doc

- Adapt inputCard.xml

Set the pathes in tags O2general/mainDir/[O2local, O2Physicslocal] to the actual O2 installation path, e.g. /home/me/alice/[O2, O2Physics] unless you run from the directory scripts/datamodel-doc. The other parameters should fit.

- Run it

./extractDataModel.py > htmloutput.txt

- Update the markdown files with the content of htmloutput.txt.


### Update the markdown files automatically

The python script mdUpdate.py allows to update the contents of the md files automatically.

mdUpdate.py takes four arguments:

Usage:
mdUpdate.py cc fn2u fnold fnnew

cc: 1: AO2D, 2: Helpers, 3: PWG tables, 4: Joins

fn2u: file with new text

fnold: file with old text

fnnew: file with replaced text

mdUpdate.py replaces in file fnold the block of text which is delimited by two lines containing a delimiter string. The block of text is replaced by the block of text in file fn2u which is delimited by two lines containing the same delimiter string. The resulting text is written to file fnnew. The delimiter string is obtained from the inputCard.xml, depending on the value of cc. If fnnew = fnold, the content of fnold is overwritten.

So to update the md files do:

- ./extractDataModel.py > htmloutput.txt
- path2mds=./testing
- ./mdUpdate.py 1 htmloutput.txt $path2mds/ao2dTables.md $path2mds/ao2dTables.md
- ./mdUpdate.py 2 htmloutput.txt $path2mds/helperTaskTables.md $path2mds/helperTaskTables.md
- ./mdUpdate.py 3 htmloutput.txt $path2mds/helperTaskTables.md $path2mds/pwgTables.md
- ./mdUpdate.py 4 htmloutput.txt $path2mds/joinsAndIterators.md $path2mds/joinsAndIterators.md

### For a full automatic update

In the same directory have O2 cloned to O2 and the documentation (your fork of https://github.com/AliceO2Group/analysis-framework) in analysis-framework and execute

O2/scripts/datamodel-doc/update-datamodel.sh
