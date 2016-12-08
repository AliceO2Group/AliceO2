###Generating Doxygen documentation

This directory contains the necessary files to automatically generate documentation for the AliceO2 project using 
Doxygen. To create the documentation, set the flag -DBUILD_DOXYGEN=ON when calling cmake; the doxygen documentation 
will then be generated when calling make.  The generated html files can be found in the "doxygen/doc/html" subdirectory 
of the build directory.

Doxygen documentation is also available online [here](http://aliceo2group.github.io/AliceO2/)
