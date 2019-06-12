# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "FrameworkLogger")
set(MODULE_BUCKET_NAME O2FrameworkLogger_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS src/Logger.cxx)
set(HEADERS
  include/Framework/Logger.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME O2FrameworkLogger_bucket
  TEST_SRCS test/unittest_Logger.cxx
)

install(FILES ${HEADERS} DESTINATION include/Framework)
