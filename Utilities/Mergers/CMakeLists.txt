# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See https://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "Mergers")

# todo create bucket
set(MODULE_BUCKET_NAME mergers_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Merger.cxx
  src/MergerInfrastructureBuilder.cxx
  src/MergerCache.cxx
  src/MergerBuilder.cxx
  )

set(HEADERS
  include/Mergers/Merger.h
  include/Mergers/MergerConfig.h
  include/Mergers/MergeInterface.h
  include/Mergers/MergeInterfaceOverrideExample.h
  include/Mergers/MergerInfrastructureBuilder.h
  include/Mergers/MergerBuilder.h
  include/Mergers/MergerCache.h
  )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})
set(LINKDEF include/Mergers/LinkDef.h)

O2_GENERATE_LIBRARY()

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_EXECUTABLE(
  EXE_NAME "mergersTopologyExample"
  SOURCES "src/mergersTopologyExample.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

O2_GENERATE_EXECUTABLE(
  EXE_NAME "mergersBenchmarkTopology"
  SOURCES "src/mergersBenchmarkTopology.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

set(TEST_SRCS
  test/test_InfrastructureBuilder.cxx
  )

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)