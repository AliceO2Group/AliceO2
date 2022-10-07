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

include_guard()

#
# o2_add_dpl_workflow(basename SOURCES ...) add a new dpl workflow executable.
# Besides building the executable, this will also generate a configuration json
# file via <executable-name> --dump so that it can be easily registered and
# deployed e.g. in the train infrastructure.
#
# For most of the options please see how o2_add_executable works.
#
# The installed executable will be named as regular executables (see
# o2_add_executable for details)
#
# The installed JSON file will be named <executable-name>.json and installed
# under share/dpl directory
#

function(o2_add_dpl_workflow baseTargetName)

  cmake_parse_arguments(PARSE_ARGV 1 A "" "COMPONENT_NAME;TARGETVARNAME"
                        "SOURCES;PUBLIC_LINK_LIBRARIES;JOB_POOL")

  if(A_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Got trailing arguments ${A_UNPARSED_ARGUMENTS}")
  endif()

  o2_add_executable(${baseTargetName}
    COMPONENT_NAME ${A_COMPONENT_NAME} TARGETVARNAME targetExeName
    SOURCES ${A_SOURCES}
    PUBLIC_LINK_LIBRARIES O2::Framework ${A_PUBLIC_LINK_LIBRARIES})

  if(A_TARGETVARNAME)
    set(${A_TARGETVARNAME}
        ${targetExeName}
        PARENT_SCOPE)
  endif()
  if(A_JOB_POOL)
    set_property(TARGET ${targetExeName} PROPERTY JOB_POOL_COMPILE ${A_JOB_POOL})
  endif()

  set(jsonFile $<TARGET_FILE_BASE_NAME:${targetExeName}>.json)

  add_custom_command(
    TARGET ${targetExeName} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env ASAN_OPTIONS=detect_leaks=0,detect_container_overflow=0,detect_odr_violation=0 "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$$LD_LIBRARY_PATH" $<TARGET_FILE:${targetExeName}> -b --dump-workflow --dump-workflow-file ${jsonFile})
  add_dependencies(${targetExeName} O2::FrameworkAnalysisSupport O2::FrameworkPhysicsSupport O2::FrameworkCCDBSupport)

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${jsonFile}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/dpl)

endfunction()
