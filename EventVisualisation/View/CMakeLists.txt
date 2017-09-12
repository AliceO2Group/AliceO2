# @author  Jeremi Niedziela

set(MODULE_NAME "EventVisualisationView")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/MultiView.cxx
    src/Initializer.cxx
    )

# HEADERS is not needed if we don't generate a dictionary.
set(HEADERS
    include/${MODULE_NAME}/MultiView.h
    include/${MODULE_NAME}/Initializer.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME event_visualisation_view_bucket)

O2_GENERATE_LIBRARY()

# Define application
O2_GENERATE_EXECUTABLE(
EXE_NAME o2eve
SOURCES src/main.cxx
MODULE_LIBRARY_NAME ${LIBRARY_NAME}
BUCKET_NAME ${BUCKET_NAME}
)
