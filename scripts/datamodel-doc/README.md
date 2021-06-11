# ALICEO2dataModel converter

Allows to create an html representation of the ALICE O2 data model.
The generated html code can be inserted in docs/datamodel/[ao2dTables, helperTaskTables, joinsAndIterators].md of https://github.com/AliceO2Group/analysis-framework/.

## Internals

The ALICE O2 Data Analysis Framework is based on a number of flat tables which contain the experimental data. The tables are declared in several header files. Some of the tables are filled automatically when an AO2D file is processed others are filled by specific tasks.

The ALICEO2dataModel converter analyses these header and task code files and extracts the table and column definitions of the tables. The information is converted into a html representation which can be included in the documentation site of the O2 Analysis Framework at https://github.com/AliceO2Group/analysis-framework/.

The converter is implemented in python. The parsing functionality is contained in ALICEO2dataModel.py and the main program is in extractDataModel.py. The process is configured with inputCard.xml.


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

- Produces

Tables have a producer. In case of the tables filled with data from the AO2D files the producer is defined to be AO2D. For tables which are filled by analysis tasks the producer is deduced from the related Produces function declaration. The code files are scanned for Produces declarations and the name of the code file is added to the respective table in the previously described 'datamodel' object.

#### CMakeLists.txt files
CMakeLists.txt files are combed for definitions of:

- o2_add_dpl_workflow (executables and the related code files)

The CMakeLists files are finally needed to relate a given code file with an executable, which allows to associate a table with its producing executable.

### Output

The html representation of the 'datamodel' object is achieved with the printHTML method.


## As simple as that

- Install the software

git clone [git@github.com:pbuehler/ALICEO2dataModel.git](git@github.com:pbuehler/ALICEO2dataModel.git)

- Adapt inputCard.xml

Set the path in tag data/O2general/mainDir/local to the actual O2 installation path, e.g. home/me/alice/O2. The other parameters should fit.

- Run it

./extractDataModel.py > htmloutput.txt

- Update the markdown files with the content of htmloutput.txt.
