# @author  Jeremi Niedziela

set(MODULE_NAME "EventVisualisationBase")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/ConfigurationManager.cxx
    src/DataInterpreter.cxx
    src/EventManager.cxx
    src/GeometryManager.cxx
    src/Track.cxx
    )

# HEADERS is not needed if we don't generate a dictionary.
set(HEADERS
    include/${MODULE_NAME}/ConfigurationManager.h
    include/${MODULE_NAME}/DataInterpreter.h
    include/${MODULE_NAME}/EventManager.h
    include/${MODULE_NAME}/GeometryManager.h
    include/${MODULE_NAME}/Track.h
    include/${MODULE_NAME}/VisualisationConstants.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME event_visualisation_base_bucket)

O2_GENERATE_LIBRARY()
