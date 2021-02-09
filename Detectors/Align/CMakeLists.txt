# **************************************************************************
# * Copyright(c) 1998-2014, ALICE Experiment at CERN, All rights reserved. *
# *                                                                        *
# * Author: The ALICE Off-line Project.                                    *
# * Contributors are mentioned in the code where appropriate.              *
# *                                                                        *
# * Permission to use, copy, modify and distribute this software and its   *
# * documentation strictly for non-commercial purposes is hereby granted   *
# * without fee, provided that the above copyright notice appears in all   *
# * copies and that both the copyright notice and this permission notice   *
# * appear in the supporting documentation. The authors make no claims     *
# * about the suitability of this software for any purpose. It is          *
# * provided "as is" without express or implied warranty.                  *
# **************************************************************************

# Module
set(MODULE ALIGN)
add_definitions(-D_MODULE_="${MODULE}")

# Module include folders
include_directories(${AliRoot_SOURCE_DIR}/${MODULE})

# Additional include directories - alphabetical order except ROOT
include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
include_directories(${AliRoot_SOURCE_DIR}/STEER/STEERBase
                    ${AliRoot_SOURCE_DIR}/STEER/STEER
		    ${AliRoot_SOURCE_DIR}/STEER/ESD
		    ${AliRoot_SOURCE_DIR}/STEER/CDB
                    ${AliRoot_SOURCE_DIR}/TRD/TRDbase
                    ${AliRoot_SOURCE_DIR}/ITS/ITSbase
                    ${AliRoot_SOURCE_DIR}/TOF/TOFbase
		    ${AliRoot_SOURCE_DIR}/HMPID/HMPIDbase
		   )

# Sources - alphabetical order
set(SRCS
  AliAlgAux.cxx
  AliAlgConstraint.cxx
  AliAlgDet.cxx
  AliAlgDetHMPID.cxx
  AliAlgDetITS.cxx
  AliAlgDetTOF.cxx
  AliAlgDetTPC.cxx
  AliAlgDetTRD.cxx
  AliAlgDOFStat.cxx
  AliAlgMPRecord.cxx
  AliAlgPoint.cxx
  AliAlgRes.cxx
  AliAlgResFast.cxx
  AliAlgSens.cxx
  AliAlgSensHMPID.cxx
  AliAlgSensITS.cxx
  AliAlgSensTOF.cxx
  AliAlgSensTPC.cxx
  AliAlgSensTRD.cxx
  AliAlgSteer.cxx
  AliAlgTrack.cxx
  AliAlgVol.cxx
  AliAlgVtx.cxx
  Mille.cxx
  )

# Headers from sources
string(REPLACE ".cxx" ".h" HDRS "${SRCS}")

# Generate the dictionary
# It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
get_directory_property(incdirs INCLUDE_DIRECTORIES)
generate_dictionary("${MODULE}" "${MODULE}LinkDef.h" "${HDRS}" "${incdirs}")

set(ROOT_DEPENDENCIES Core Geom Gpad Graf3d MathCore Matrix Hist RIO Tree)
set(ALIROOT_DEPENDENCIES STEERBase ESD CDB ITSbase TRDbase TOFbase)

# Generate the ROOT map
# Dependecies
set(LIBDEPS ${ALIROOT_DEPENDENCIES} ${ROOT_DEPENDENCIES})
generate_rootmap("${MODULE}" "${LIBDEPS}" "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}LinkDef.h")

# Create an object to be reused in case of static libraries 
# Otherwise the sources will be compiled twice
add_library(${MODULE}-object OBJECT ${SRCS} G__${MODULE}.cxx)
# Add a library to the project using the object
add_library(${MODULE} SHARED $<TARGET_OBJECTS:${MODULE}-object>)

target_link_libraries(${MODULE} ${ALIROOT_DEPENDENCIES} ${ROOT_DEPENDENCIES})

# Public include folders that will be propagated to the dependecies
target_include_directories(${MODULE} PUBLIC ${incdirs})

# System dependent: Modify the way the library is build
if(${CMAKE_SYSTEM} MATCHES Darwin)
    set(MODULE_LINK_FLAGS "-undefined dynamic_lookup ${MODULE_LINK_FLAGS}")
endif(${CMAKE_SYSTEM} MATCHES Darwin)

# Aditional compile and link flags
set_target_properties(${MODULE}-object PROPERTIES COMPILE_FLAGS "${MODULE_COMPILE_FLAGS}")
set_target_properties(${MODULE} PROPERTIES LINK_FLAGS "${MODULE_LINK_FLAGS}")

# Setting the correct headers for the object as gathered from the dependencies
# After all compilation flags are set
target_include_directories(${MODULE}-object PUBLIC $<TARGET_PROPERTY:${MODULE},INCLUDE_DIRECTORIES>)
set_target_properties(${MODULE}-object PROPERTIES COMPILE_DEFINITIONS $<TARGET_PROPERTY:${MODULE},COMPILE_DEFINITIONS>)

# Installation
install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# install macros
#install(DIRECTORY utils DESTINATION  ALIGN)

install(FILES ${HDRS} DESTINATION include)

