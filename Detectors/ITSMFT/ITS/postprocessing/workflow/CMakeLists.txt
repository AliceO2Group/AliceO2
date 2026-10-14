# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_executable(postprocessing-workflow
                  COMPONENT_NAME its-standalone
                  TARGETVARNAME postprocessingWorkflowTarget
                  SOURCES standalone-postprocessing-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ITSPostprocessing)

if(BUILD_TESTING)
  # Help constructs the workflow and exits, exercising static initialization and
  # destruction without input data or CCDB access.
  set(helpTestDirectory "${CMAKE_CURRENT_BINARY_DIR}/test-workflow-help")
  set(beambkgHelpTestDirectory "${CMAKE_CURRENT_BINARY_DIR}/test-workflow-beambkg-help")
  file(MAKE_DIRECTORY "${helpTestDirectory}" "${beambkgHelpTestDirectory}")
  add_test(NAME its-standalone-postprocessing-help
           COMMAND $<TARGET_FILE:${postprocessingWorkflowTarget}> --disable-mc --help)
  add_test(NAME its-standalone-postprocessing-beambkg-help
           COMMAND $<TARGET_FILE:${postprocessingWorkflowTarget}>
                   --disable-mc --input-from-upstream --its-beambkg-study
                   --cluster-sources ITS --track-sources "" --help)
  set_tests_properties(its-standalone-postprocessing-help
                       its-standalone-postprocessing-beambkg-help PROPERTIES
                       LABELS "its;workflow;itsmft"
                       TIMEOUT 60)
  set_tests_properties(its-standalone-postprocessing-help PROPERTIES
                       WORKING_DIRECTORY "${helpTestDirectory}")
  set_tests_properties(its-standalone-postprocessing-beambkg-help PROPERTIES
                       WORKING_DIRECTORY "${beambkgHelpTestDirectory}")
endif()
