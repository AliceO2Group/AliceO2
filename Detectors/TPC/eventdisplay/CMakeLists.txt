set(MODULE_NAME "TPCEventdisplay")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/HitDraw.cxx
)

set(HEADERS
   include/${MODULE_NAME}/HitDraw.h
)
Set(LINKDEF src/TPCEventdisplayLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_eventdisplay_bucket)

O2_GENERATE_LIBRARY()
