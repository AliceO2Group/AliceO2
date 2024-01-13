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

# file helpers.cmake
# author David Rohr

function(create_binary_resource RESOURCE OUTPUTFILE)
  get_filename_component(input-file-abs ${RESOURCE} ABSOLUTE)
  FILE(RELATIVE_PATH input-file-rel ${CMAKE_CURRENT_BINARY_DIR} ${input-file-abs})
  add_custom_command(
    OUTPUT ${OUTPUTFILE}
    COMMAND ${CMAKE_LINKER} --relocatable --format binary --output ${OUTPUTFILE} ${input-file-rel}
    DEPENDS ${RESOURCE}
    COMMENT "Adding binary resource ${input-file-rel}"
    VERBATIM
  )
endfunction(create_binary_resource)

function(add_binary_resource TARGET RESOURCE)
    set(output-file ${RESOURCE}.o)
    create_binary_resource(${RESOURCE} ${output-file})
    target_sources(${TARGET} PRIVATE ${output-file})
endfunction(add_binary_resource)
