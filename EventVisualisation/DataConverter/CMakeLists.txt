# @author  Jeremi Niedziela

set(MODULE_NAME "EventVisualisationDataConverter")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/MinimalisticEvent.cxx
    src/MinimalisticTrack.cxx
)

# HEADERS is not needed if we don't generate a dictionary.
set(HEADERS
    include/${MODULE_NAME}/MinimalisticEvent.h
    include/${MODULE_NAME}/MinimalisticTrack.h
    include/${MODULE_NAME}/ConversionConstants.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME event_visualisation_data_converter_bucket)

O2_GENERATE_LIBRARY()
