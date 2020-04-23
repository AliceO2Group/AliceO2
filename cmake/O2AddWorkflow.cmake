# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

include_guard()

#
# o2_add_dpl_workflow(basename SOURCES ...) add a new dpl workflow
# executable. Besides building the executable, this will also generate
# a configuration json file via <executable-name> --dump so that it can
# be easily registered and deployed e.g. in the train infrastructure.
#
# For most of the options please see how o2_add_executable works.
#
# The installed executable will be named o2[-exeType][-component_name]-basename
# The installed JSON file will be named o2-[exeType][-component_name]-basename.json.
#

function(o2_add_dpl_workflow baseTargetName)

  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        ""
                        "COMPONENT_NAME;TARGETVARNAME"
                        "SOURCES;PUBLIC_LINK_LIBRARIES")

  if(A_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Got trailing arguments ${A_UNPARSED_ARGUMENTS}")
  endif()

  if(A_COMPONENT_NAME)
    string(TOLOWER ${A_COMPONENT_NAME} component)
    set(comp -${component})
  endif()
  set(exeName o2${exeType}${comp}-${baseTargetName})

  o2_add_executable(${baseTargetName} 
                    COMPONENT_NAME ${A_COMPONENT_NAME}
                    TARGETVARNAME ${A_TARGETVARNAME}
                    SOURCES ${A_SOURCES}
                    PUBLIC_LINK_LIBRARIES O2::Framework ${A_PUBLIC_LINK_LIBRARIES})
  add_custom_command(OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${exeName}.json 
                     DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${exeName}
                     COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${exeName} | cat > ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${exeName}.json
                    )
  
  add_custom_target(${exeName}_dpl_config 
                    DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${exeName}.json
                   )
  install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${exeName}.json
          DESTINATION ${CMAKE_INSTALL_DATADIR}/dpl
  )
endfunction()
