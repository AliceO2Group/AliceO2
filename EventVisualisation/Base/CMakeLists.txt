# @author  Jeremi Niedziela

set(MODULE_NAME "EventVisualisationBase")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/EventManager.cxx
    src/Initializer.cxx
    src/MultiView.cxx
    src/GeometryManager.cxx
    )

# HEADERS is not needed if we don't generate a dictionary.
set(HEADERS
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME event_visualisation_bucket)

O2_GENERATE_LIBRARY()

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME o2eve
    SOURCES src/main.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)
