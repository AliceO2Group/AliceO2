# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# @author Sandro Wenzel @brief  cmake setup for module Utilities/MCStepLogger

set(MODULE_NAME "MCStepLogger")
set(MODULE_BUCKET_NAME mcsteplogger_bucket)
o2_setup(NAME ${MODULE_NAME})

set(SRCS
    src/MCStepInterceptor.cxx
    src/MCStepLoggerImpl.cxx
    src/StepInfo.cxx
    src/MCAnalysis.cxx
    src/BasicMCAnalysis.cxx
    src/MCAnalysisManager.cxx
    src/MCAnalysisFileWrapper.cxx
    src/MCAnalysisUtilities.cxx
    src/ROOTIOUtilities.cxx)

set(HEADERS
    include/${MODULE_NAME}/StepInfo.h
    include/${MODULE_NAME}/MetaInfo.h
    include/${MODULE_NAME}/MCAnalysis.h
    include/${MODULE_NAME}/BasicMCAnalysis.h
    include/${MODULE_NAME}/MCAnalysisManager.h
    include/${MODULE_NAME}/MCAnalysisFileWrapper.h
    include/${MODULE_NAME}/MCAnalysisUtilities.h
    include/${MODULE_NAME}/ROOTIOUtilities.h)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})
set(LINKDEF src/MCStepLoggerLinkDef.h)

o2_generate_library()

o2_generate_executable(EXE_NAME
                       mcStepAnalysis
                       SOURCES
                       src/analyseMCSteps.cxx
                       MODULE_LIBRARY_NAME
                       ${LIBRARY_NAME}
                       BUCKET_NAME
                       ${BUCKET_NAME})

o2_generate_executable(EXE_NAME
                       runTestBasicMCAnalysis
                       SOURCES
                       src/basicMCAnalysisCI.cxx
                       MODULE_LIBRARY_NAME
                       ${LIBRARY_NAME}
                       BUCKET_NAME
                       ${BUCKET_NAME})

# check correct functioning of the logger and the MC analysis chain
if(BUILD_SIMULATION)
  add_test_wrap(NAME
                mcloggertest
                COMMAND
                ${CMAKE_BINARY_DIR}/bin/o2-sim-tpc
                -n
                1
                -e
                TGeant3)
  # tests if the logger was active
  set_tests_properties(mcloggertest
                       PROPERTIES PASS_REGULAR_EXPRESSION "VolName.*COUNT")
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(PRELOAD "DYLD_INSERT_LIBRARIES")
  else()
    set(PRELOAD "LD_PRELOAD")
  endif()
  set_tests_properties(
    mcloggertest
    PROPERTIES
      ENVIRONMENT
      ${PRELOAD}=${CMAKE_BINARY_DIR}/lib/libMCStepLogger${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
  set_property(TEST mcloggertest
               APPEND
               PROPERTY ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR})

  # check whether StepLogger output can be written to ROOT file fix output file
  # name for logged data
  set(STEPLOGGER_ROOTFILE "MCStepLoggerOutput_test.root")
  add_test_wrap(NAME
                mcloggertest_tofile
                COMMAND
                ${CMAKE_BINARY_DIR}/bin/runTPC
                -n
                1
                -e
                TGeant3)
  set_tests_properties(
    mcloggertest_tofile
    PROPERTIES
      ENVIRONMENT
      ${PRELOAD}=${CMAKE_BINARY_DIR}/lib/libMCStepLogger${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
  # set environment accordingly
  set_property(TEST mcloggertest_tofile
               APPEND
               PROPERTY ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR}
                        MCSTEPLOG_TTREE=1
                        MCSTEPLOG_OUTFILE=${STEPLOGGER_ROOTFILE})

  # check for working analysis fix output file name for analysis
  set(MCANALYSIS_ROOTFILE "BasicMCAnalysis.root")
  add_test_wrap(NAME
                basicmcanalysis
                COMMAND
                ${CMAKE_BINARY_DIR}/bin/mcStepAnalysis
                analyze
                -f
                ${STEPLOGGER_ROOTFILE}
                -o
                ${MCANALYSIS_ROOTFILE}
                -l
                testLabel)
  set_tests_properties(basicmcanalysis PROPERTIES DEPENDS mcloggertest_tofile)
  # set environment accordingly
  set_property(TEST basicmcanalysis
               APPEND
               PROPERTY ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR})

endif()
