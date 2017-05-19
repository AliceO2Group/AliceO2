set(MODULE_NAME "DetectorsBaseTest")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
)

set(HEADERS
  )

# did not manage to run with this dictionary, need to move content
# to DetectorsBaseLinkDef
#Set(LINKDEF testDetBaseLinkDef.h)
Set(LINKDEF src/BaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME detectors_base)

O2_GENERATE_LIBRARY()
