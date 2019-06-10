o2_add_library(MCHMappingImpl3
               SOURCES src/GenDetElemId2SegType.cxx
                       src/GenDetElemId2SegType.h
                       src/PadGroup.h
                       src/PadGroupType.cxx
                       src/PadGroupType.h
                       src/CathodeSegmentationCImpl3.cxx
                       src/CathodeSegmentationCreator.cxx
                       src/CathodeSegmentationCreator.h
                       src/CathodeSegmentationImpl3.cxx
                       src/CathodeSegmentationImpl3.h
               PUBLIC_LINK_LIBRARIES Boost::boost O2::MCHMappingInterface ms_gsl
               TARGETVARNAME targetName)

# We add all segmentation creators by default, but the final goal would be to
# tailor this for each executable reaching a given FLP (so it gets only the
# mapping it needs)
foreach(segtype RANGE 20)
  target_sources(${targetName} PRIVATE
                 src/GenCathodeSegmentationCreatorForSegType${segtype}.cxx)
endforeach()

include(GenerateExportHeader)
generate_export_header(${targetName} BASE_NAME o2mchmappingimpl3)

set_target_properties(${targetName} PROPERTIES CXX_VISIBILITY_PRESET hidden)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(${targetName} PRIVATE -fext-numeric-literals)
endif()
if(APPLE)
  add_custom_command(
    TARGET ${targetName} POST_BUILD
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/../../../check_nof_exported_symbols.sh
            $<TARGET_LINKER_FILE:MCHMappingImpl3> 18
    COMMENT "Checking number of exported symbols in the library")
endif()
