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

# file vulkan_display.cmake
# author David Rohr

if (GLSLC STREQUAL "GLSLC-NOTFOUND")
    message(FATAL_ERROR "glslc not found")
endif()

function(add_glslc_shader TARGET SHADER)
    get_filename_component(input-file-abs ${SHADER} ABSOLUTE)
    FILE(RELATIVE_PATH input-file-rel ${CMAKE_CURRENT_SOURCE_DIR} ${input-file-abs})
    set(spirv-file ${CMAKE_CURRENT_BINARY_DIR}/shaders/${input-file-rel}.spv)
    get_filename_component(output-dir ${spirv-file} DIRECTORY)
    file(MAKE_DIRECTORY ${output-dir})

    add_custom_command(
        OUTPUT ${spirv-file}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${spirv-file} ${input-file-abs}
        DEPENDS ${input-file-abs}
        IMPLICIT_DEPENDS CXX ${input-file-abs}
        COMMENT "Compiling GLSL to SPIRV: ${SHADER}"
        VERBATIM
    )

    add_binary_resource(${TARGET} ${spirv-file})
endfunction(add_glslc_shader)
