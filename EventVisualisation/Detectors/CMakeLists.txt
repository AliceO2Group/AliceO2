# @author  Jeremi Niedziela

set(MODULE_NAME "EventVisualisationDetectors")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/DataInterpreterRND.cxx
    )

# HEADERS is not needed if we don't generate a dictionary.
set(HEADERS
    include/${MODULE_NAME}/DataInterpreterRND.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME event_visualisation_detectors_bucket)

O2_GENERATE_LIBRARY()
