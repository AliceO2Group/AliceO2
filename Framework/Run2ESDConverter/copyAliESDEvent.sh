#!/bin/bash -ex

#SEEDS="AliESDEvent AliExternalTrackParam AliVTrack AliVEvent AliTOFHeader AliESDRun AliTimeStamp AliESDHeader AliDigit AliBitPacking AliESDZDC AliVZDC AliDigitizationInput AliStream AliFMDFloatMap AliPIDResponse AliSimulation"
SEEDS="AliESDEvent TTreeStream"
mkdir -p src

copy_seed() {
  local s=$1
  find ../../../AliRoot/ -name $s.h -exec cp {} src/ \;
  find ../../../AliRoot/ -name $s.cxx -exec cp {} src/ \;
  MOREFILES=$(cat src/$s.cxx | grep -e "#[ ]*include \"" | sed -e's/.* "//;s/".*//g' | grep -v "^T" | sed -e 's/.h$//')
  MOREFILES="$MOREFILES $(cat src/$s.cxx | grep -e "#[ ]*include <Ali" | sed -e's/.* <//;s/>.*//g' | grep -v "^T" | sed -e 's/.h$//')"
  MOREFILES="$MOREFILES $(cat src/$s.h | grep -e "#[ ]*include \"" | sed -e's/.* "//;s/"//g' | sed -e 's/.h$//')"
  MOREFILES="$MOREFILES $(cat src/$s.h | grep -e "#[ ]*include <Ali" | sed -e's/.* <//;s/>//g' | sed -e 's/.h$//')"
  echo $MOREFILES
  for m in $MOREFILES; do
    if [ -f src/$m.h ]; then
      continue
    fi
    if [ -f src/$m.cxx ]; then
      continue
    fi
    copy_seed $m
#    find ../../../AliRoot/ -name $m.h -exec cp {} src/ \;
#    find ../../../AliRoot/ -name $m.cxx -exec cp {} src/ \;
    SRCS="$SRCS $m.cxx"
    HDRS="$HDRS $m.h"
  done
}

for s in $SEEDS ; do
  copy_seed $s
done

cat << \EOF >CMakeLists.txt
# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "Run2ESDConverter")
set(MODULE_BUCKET_NAME FrameworkApplication_bucket)

O2_SETUP(NAME ${MODULE_NAME})
set(SRCS
  @SRCS@
   )

set(HEADERS
  @HDRS@
  )

## TODO: feature of macro, it deletes the variables we pass to it, set them again
## this has to be fixed in the macro implementation
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})
set(LINKDEF src/Run2ConversionLinkdef.h)

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME "o2AODRun2ESDReader"
  SOURCES "src/o2AODRun2ESDReader.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)
EOF

LINKDEFS="ESD STEERBase STEER HLTbase CDB STAT RAWDatarec RAWDatabase HMPIDbase HMPIDrec RAWDatasim"
for x in $LINKDEFS; do
  find ../../../AliRoot/ -name ${x}LinkDef.h -exec cp {} src/ \;
done

cat << \EOF > src/Run2ConversionLinkdef.h
#include "ESDLinkDef.h"
#include "STEERBaseLinkDef.h"
#include "STEERLinkDef.h"
#include "HLTbaseLinkDef.h"
#include "CDBLinkDef.h"
#include "STATLinkDef.h"
#include "RAWDatarecLinkDef.h"
#include "RAWDatabaseLinkDef.h"
#include "HMPIDbaseLinkDef.h"
#include "HMPIDrecLinkDef.h"
#include "RAWDatasimLinkDef.h"
EOF

SRCS=`find src -name *.cxx | grep -v AliHLT`
HDRS=`find src -name *.h | grep -v AliHLT | grep -i -v LinkDef.h | grep -v Run2ESDConversionHelpers.h`
echo $SRCS
echo $HDRS
perl -p -i -e "s|[\@]SRCS[\@]|${SRCS}|" CMakeLists.txt
perl -p -i -e "s|[\@]HDRS[\@]|${HDRS}|" CMakeLists.txt
perl -p -i -e 's/<AliTimeStamp.h>/"AliTimeStamp.h"/' src/AliESDRun.h
perl -p -i -e 's/<AliVZDC.h>/"AliVZDC.h"/' src/AliESDZDC.h
rm -fr src/AliHLT*
#ed src/AliSimulation.cxx << \EOF
#,s/.*AliHLTSimulation.h.*//
#/AliSimulation::CreateHLT/+2
#/AliSimulation::CreateHLT/+2,/^}/-1d
#/AliSimulation::RunHLT/+2
#/AliSimulation::RunHLT/+2,/^}/-1d
#wq
#EOF
exit 1
cd ../../../sw/BUILD/O2-latest/O2/Framework/Run2ESDConverter
make -j 20 -k
