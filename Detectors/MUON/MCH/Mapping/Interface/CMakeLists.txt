O2_SETUP(NAME MCHMappingInterface)
set(BUCKET_NAME mch_mapping_interface_bucket)

add_library(MCHMappingInterface INTERFACE)

install(DIRECTORY include/MCHMappingInterface DESTINATION include)

# custom target so that some IDE (e.g. CLion) recognize those files as
# being part of the project even though they are not part of a CMake target
add_custom_target(MCHMappingInterface_ide SOURCES
        include/MCHMappingInterface/Segmentation.h
        include/MCHMappingInterface/SegmentationCInterface.h
        )
