# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
# 
# See https://alice-o2.web.cern.ch/ for full licensing information.
# 
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "Framework")

set(MODULE_BUCKET_NAME O2DeviceApplication_bucket)

O2_SETUP(NAME ${MODULE_NAME})
if (GLFW_FOUND)
  set(GUI_SOURCES src/FrameworkGUIDebugger.cxx)
else()
  set(GUI_SOURCES src/FrameworkDummyDebugger.cxx src/DummyDebugGUI.cxx)
endif()

set(SRCS
    src/DataAllocator.cxx
    src/DataProcessingDevice.cxx
    src/DataProcessor.cxx
    src/DataRelayer.cxx
    src/DataSourceDevice.cxx
    src/DeviceMetricsInfo.cxx
    src/GraphvizHelpers.cxx
    src/SimpleMetricsService.cxx
    src/runDataProcessing.cxx
    ${GUI_SOURCES}
   )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

# TODO: feature of macro, it deletes the variables we pass to it, set them again
# this has to be fixed in the macro implementation
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_EXECUTABLE(
  EXE_NAME "test_SimpleDataProcessingDevice01"
  SOURCES "test/test_SimpleDataProcessingDevice01.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

target_compile_options(Framework PUBLIC -O0 -g -fno-omit-frame-pointer)
target_compile_options(test_SimpleDataProcessingDevice01 PUBLIC -O0 -g -fno-omit-frame-pointer)

set(TEST_SRCS test/test_Variants.cxx
              test/test_Collections.cxx
              test/test_Services.cxx
              test/test_DeviceMetricsInfo.cxx)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
