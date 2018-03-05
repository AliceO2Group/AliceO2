O2_SETUP(NAME MCHMappingImpl3)
set(BUCKET_NAME mch_mapping_impl3_bucket)

set(SRCS
        src/GenDetElemId2SegType.cxx
        src/GenDetElemId2SegType.h
        src/PadGroup.h
        src/PadGroupType.cxx
        src/PadGroupType.h
        src/SegmentationCImpl3.cxx
        src/SegmentationCreator.cxx
        src/SegmentationCreator.h
        src/SegmentationImpl3.cxx
        src/SegmentationImpl3.h
        )

# We add all segmentation creators by default,
# but the final goal would be to tailor this
# for each executable reaching a given FLP (so it gets only
# the mapping it needs)
foreach (segtype RANGE 20)
    set(SRCS ${SRCS} src/GenSegmentationCreatorForSegType${segtype}.cxx)
endforeach ()

set(LIBRARY_NAME MCHMappingImpl3)
O2_GENERATE_LIBRARY()

include(GenerateExportHeader)
generate_export_header(MCHMappingImpl3)

set_target_properties(MCHMappingImpl3 PROPERTIES CXX_VISIBILITY_PRESET hidden)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(MCHMappingImpl3 PRIVATE -fext-numeric-literals)
endif ()
if (APPLE)
    add_custom_command(TARGET MCHMappingImpl3 POST_BUILD
            COMMAND ${CMAKE_SOURCE_DIR}/Detectors/MUON/check_nof_exported_symbols.sh $<TARGET_LINKER_FILE:MCHMappingImpl3> 18
            COMMENT "Checking number of exported symbols in the library")
endif ()

