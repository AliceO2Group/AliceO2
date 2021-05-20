# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHMappingImpl4
        SOURCES src/CreateSegmentation.cxx
        src/GenDetElemId2SegType.cxx
        src/GenDetElemId2SegType.h
        src/PadGroup.h
        src/PadGroupType.cxx
        src/PadGroupType.h
        src/CathodeSegmentationCImpl4.cxx
        src/CathodeSegmentationCreator.cxx
        src/CathodeSegmentationCreator.h
        src/CathodeSegmentationImpl4.cxx
        src/CathodeSegmentationImpl4.h
        PUBLIC_LINK_LIBRARIES Boost::boost O2::MCHMappingInterface
        Microsoft.GSL::GSL fmt::fmt
        TARGETVARNAME targetName)

# We add all segmentation creators by default, but the final goal would be to
# tailor this for each executable reaching a given FLP (so it gets only the
# mapping it needs)
foreach(segtype RANGE 20)
        target_sources(${targetName} PRIVATE
                src/GenCathodeSegmentationCreatorForSegType${segtype}.cxx)
endforeach()

include(GenerateExportHeader)
generate_export_header(${targetName} BASE_NAME o2mchmappingimpl4)

set_target_properties(${targetName} PROPERTIES CXX_VISIBILITY_PRESET hidden)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(${targetName} PRIVATE -fext-numeric-literals)
endif()
if(APPLE)
        get_filename_component(
                script ${CMAKE_CURRENT_LIST_DIR}/../../../check_nof_exported_symbols.sh
                ABSOLUTE)
        add_custom_command(
                TARGET ${targetName} POST_BUILD
                COMMAND ${script} $<TARGET_LINKER_FILE:${targetName}> 19
                COMMENT "Checking number of exported symbols in the library")
endif()
