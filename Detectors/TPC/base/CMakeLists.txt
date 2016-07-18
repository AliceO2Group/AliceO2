
set(INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/

)

set(SYSTEM_INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}
  ${BASE_INCLUDE_DIRECTORIES}
  ${Boost_INCLUDE_DIRS}
  ${FAIRROOT_INCLUDE_DIR}
)
include_directories( ${INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES
  ${ROOT_LIBRARY_DIR}
  ${FAIRROOT_LIBRARY_DIR}
)

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/ContainerFactory.cxx
   src/CRU.cxx
   src/DigitPos.cxx
   src/FECInfo.cxx
   src/Mapper.cxx
   src/PadInfo.cxx
   src/PadPos.cxx
   src/PadRegionInfo.cxx
   src/PadSecPos.cxx
   src/Point2D.cxx
   src/Point3D.cxx
   src/ROC.cxx
   src/Sector.cxx
)


set(HEADERS
   include/ContainerFactory.h
   include/CRU.h
#    include/Defs.h
   include/DigitPos.h
   include/FECInfo.h
   include/Mapper.h
   include/PadInfo.h
   include/PadPos.h
   include/PadRegionInfo.h
   include/PadSecPos.h
   include/Point2D.h
   include/Point3D.h
   include/ROC.h
   include/Defs.h
   include/Sector.h
)

# string(REPLACE ".cxx" ".h" HEADERS "${SRCS}")

Set(LINKDEF src/tpcBaseLinkDef.h)
Set(LIBRARY_NAME tpcBase)
Set(DEPENDENCIES
    Base
)
GENERATE_LIBRARY()
