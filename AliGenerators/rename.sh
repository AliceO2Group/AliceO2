#!/bin/bash

# The rename script exchange all occurence of PndPythia or PndPythia
# by the name given by the first parameter. If the detector is for example 
# the Trd of the Cbm experiment a good name is CbmTrd. Normaly one should
# use the naming convention of the experiment.
# Also the filenames any many more things are changed automatically. In the
# end there are only some small changes which have to be done by hand.

#set -xv

if [ $# -ne 3 ]; then
  echo "Please call the script with three parameters. The first one is the"
  echo "name of the detector. The second is the name of the project. This"
  echo "name can be found in the main CMakeLists.txt as argument for"
  echo "Project(<ProjectName>)). The third parameter is the prefix in front"
  echo "the class names. For CBM this is for example Cbm, for Panda Pnd."
  echo "If you're not sure check in already existing detectors."
  echo "The script will exchange all default names by the new name"
  exit 1
fi  

DetectorName=$1 
DetectorNameUpper=$(echo $DetectorName | tr [:lower:] [:upper:])

ProjectName=$(echo $2 | tr [:lower:] [:upper:])
ProjectSourceDir=${ProjectName}_SOURCE_DIR
RelativeDir=$(basename $PWD)
Prefix=$3
for i in $(ls PndPythia*); do 
  oldfile=$i
  newfile=$(echo $oldfile | sed "s/PndPythia/$DetectorName/")
  mv $oldfile $newfile
done 

arch=`uname -s | tr '[A-Z]' '[a-z]'`
case "$arch" in
    linux)
        sedstring="-i "
        ;;
    darwin)
        sedstring="-i .bak"
        ;;
    *)
        echo "Platform not supported"
        exit 1
        ;;
esac

find . -name "*.h" -exec sed -e "s/PndPythia/$DetectorName/g" $sedstring "{}" ";"
find . -name "*.h" -exec sed -e "s/PndPythia/$DetectorNameUpper/g" $sedstring "{}" ";"
find . -name "*.cxx" -exec sed -e "s/PndPythia/$DetectorName/g" $sedstring "{}" ";"
find . -name "*.cxx" -exec sed -e "s/PndPythia/$DetectorNameUpper/g" $sedstring "{}" ";"
find . -name "*.cxx" -exec sed -e "s/FairDetectorList/${Prefix}DetectorList/g" $sedstring "{}" ";"
find . -name "*.cxx" -exec sed -e "s/FairStack/${Prefix}Stack/g" $sedstring "{}" ";"
find . -name "*.h" -exec sed -e "s/FairDetectorList/${Prefix}DetectorList/g" $sedstring "{}" ";"
find . -name "*.h" -exec sed -e "s/FairStack/${Prefix}Stack/g" $sedstring "{}" ";"

sed -e "s#tutorial/PndPythia#$RelativeDir#g" $sedstring CMakeLists.txt
sed -e "s/PndPythia/$DetectorName/g" $sedstring CMakeLists.txt
sed -e "s/PndPythia/$DetectorNameUpper/g" $sedstring CMakeLists.txt
sed -e "s/FAIRROOT_SOURCE_DIR/$ProjectSourceDir/g" $sedstring CMakeLists.txt

if [ -d .svn ]; then  
  echo "Please remove the .svn directory."
  echo " This directory was also copied from templates."
  echo "##"
fi

echo "Please add the directories which contain the Stack and"
echo "DetectorList classes to the include directories in CMakeLists.txt"
echo "##"

echo "Please add the new detector to the detector list. This iist can be"
echo "found in the DetectorList class. The name to be added is"
echo "k${DetectorName}."

echo "##"
echo "edit ${DetectorName}Geo.h and ${DetectorName}Geo.cxx  according to the"
echo "comments in the files."

#set +xvx