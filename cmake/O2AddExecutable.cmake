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
# o2_add_executable(basename SOURCES ...) add an executable with the given
# sources.
#
# * SOURCES (required) gives the list of source files to compile into the
#   executable
# * PUBLIC_LINK_LIBRARIES (needed in most cases) indicates the list of targets
#   this executable depends on.
#
# The installed executable will be named o2[-exeType][-component_name]-basename
#
# where :
#
# * exeType is `test` if IS_TEST is set, `bench` if IS_BENCHMARK is set or void
#   otherwise
# * COMPONENT_NAME (optional) is typically used to indicate a subsystem name for
#   regular executables (e.g. o2-tpc-... or o2-mch-...) or the origin target for
#   tests (to help locate the source file in the source directory hierarchy,
#   e.g. o2-test-DataFormats-...)
# * WORKFLOW_PLUGIN (optional), for a DPL workflow: its sources are compiled
#   into the workflow plugin <Name>, in libO2<Name>Plugin, which o2-dpl-run can
#   load with --workflow-plugin, and the executable is only a main() linking
#   it. <Name> is the CamelCase of the executable name without o2-, e.g.
#   o2-tpc-reco-workflow gives O2TpcRecoWorkflowPlugin:TpcRecoWorkflow.
#
# Note that the _target_ corresponding to the executable will be named
# O2exe[-exeType][component]-basename and can be retrieved with the
# TARGETVARNAME parameter if needed.
#
# For instance after a call to:
#
# o2_add_executable(toto SOURCES ... TARGETVARNAME titi IS_TEST)
#
# ${titi} will contain something like `O2exe-test-toto` (for the exact naming
# see the o2_name_target function) and an executable named o2-test-toto will be
# created upon build)

function(o2_add_executable baseTargetName)

  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        "IS_TEST;NO_INSTALL;IS_BENCHMARK;WORKFLOW_PLUGIN"
                        "COMPONENT_NAME;TARGETVARNAME"
                        "SOURCES;PUBLIC_LINK_LIBRARIES;JOB_POOL")

  if(A_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Got trailing arguments ${A_UNPARSED_ARGUMENTS}")
  endif()

  # set the executable name following our coding convention
  if(A_IS_TEST)
    set(exeType -test)
  elseif(A_IS_BENCHMARK)
    set(exeType -bench)
  endif()

  if(A_COMPONENT_NAME)
    string(TOLOWER ${A_COMPONENT_NAME} component)
    set(comp -${component})
  endif()
  set(exeName o2${exeType}${comp}-${baseTargetName})

  if(A_IS_TEST)
    set(isTest "IS_TEST")
  endif()
  if(A_IS_BENCH)
    set(isBench "IS_BENCH")
  endif()

  # get the target name. the convention might be different from the executable
  # convention.
  o2_name_target(${baseTargetName}
                 NAME
                 targetName
                 IS_EXE
                 ${isTest}
                 ${isBench})

  set(target ${targetName})

  if(A_TARGETVARNAME)
    set(${A_TARGETVARNAME} ${target} PARENT_SCOPE)
  endif()

  # add the executable with its sources
  if(A_WORKFLOW_PLUGIN)
    o2_add_workflow_plugin(${exeName} SOURCES ${A_SOURCES}
                           PUBLIC_LINK_LIBRARIES ${A_PUBLIC_LINK_LIBRARIES}
                           TARGETVARNAME plugin)
    add_executable(${target} $<TARGET_OBJECTS:O2FrameworkWorkflowMain>)
    target_link_libraries(${target} PUBLIC ${plugin})
    if(NOT A_NO_INSTALL)
      install(TARGETS ${plugin} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
  else()
    add_executable(${target} ${A_SOURCES})
  endif()

  # set the executable output name
  set_property(TARGET ${target} PROPERTY OUTPUT_NAME ${exeName})
  if(A_JOB_POOL)
    set_property(TARGET ${target} PROPERTY JOB_POOL_COMPILE ${A_JOB_POOL})
  endif()

  if(A_IS_TEST)
    # tests go in a separate directory
    get_filename_component(outdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../tests
                           ABSOLUTE)
    set_property(TARGET ${target} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})
  endif()

  # use its dependencies
  foreach(lib IN LISTS A_PUBLIC_LINK_LIBRARIES)
    string(FIND ${lib} "::" NS)
    if(${NS} EQUAL -1)
      message(FATAL_ERROR "Trying to use a non-namespaced target ${lib}")
    endif()
    target_link_libraries(${target} PUBLIC ${lib})
  endforeach()

  # needed for fpu.c
  target_link_libraries(${target} PUBLIC ROOT::XMLIO)
  target_link_libraries(${target} PUBLIC O2::fpu_support)

  if(NOT A_NO_INSTALL)
    # install the executable

    if(A_IS_TEST)
      install(TARGETS ${target}
              RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${testsDir})
    else()
      install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()

endfunction()

# o2_add_workflow_plugin(exeName SOURCES ... PUBLIC_LINK_LIBRARIES ...
# TARGETVARNAME var) builds the workflow plugin of the executable exeName, see
# WORKFLOW_PLUGIN above, and the main() of such executables, once.
function(o2_add_workflow_plugin exeName)
  cmake_parse_arguments(PARSE_ARGV 1 A "" "TARGETVARNAME" "SOURCES;PUBLIC_LINK_LIBRARIES")
  string(REGEX REPLACE "^o2-" "" name ${exeName})
  string(REPLACE "-" ";" parts ${name})
  set(name "")
  foreach(part IN LISTS parts)
    string(SUBSTRING ${part} 0 1 first)
    string(SUBSTRING ${part} 1 -1 rest)
    string(TOUPPER ${first} first)
    string(APPEND name ${first}${rest})
  endforeach()
  string(MAKE_C_IDENTIFIER ${name} name)
  set(plugin O2${name}Plugin)
  add_library(${plugin} SHARED ${A_SOURCES})
  target_link_libraries(${plugin} PUBLIC ${A_PUBLIC_LINK_LIBRARIES})
  target_compile_definitions(${plugin} PRIVATE DPL_WORKFLOW_PLUGIN_NAME=${name})
  set(${A_TARGETVARNAME} ${plugin} PARENT_SCOPE)
  if(NOT TARGET O2FrameworkWorkflowMain)
    file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/O2FrameworkWorkflowMain.cxx CONTENT [[
extern "C" int dpl_workflow_main(int argc, char** argv);
int main(int argc, char** argv) { return dpl_workflow_main(argc, argv); }
]])
    add_library(O2FrameworkWorkflowMain OBJECT ${CMAKE_BINARY_DIR}/O2FrameworkWorkflowMain.cxx)
  endif()
endfunction()
