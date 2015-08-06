# **************************************************************************
# * Copyright(c) 1998-2015, ALICE Experiment at CERN, All rights reserved. *
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

if(DDS_PATH)
  if(EXISTS ${DDS_PATH}/lib AND EXISTS ${DDS_PATH}/include)
    set(DDS_FOUND TRUE)
    message(STATUS "DDS found at ${DDS_PATH}")
  else()
    set(DDS_FOUND FALSE)
    message(STATUS "DDS not found")
  endif()
else(DDS_PATH)
  if(EXISTS ${SIMPATH}/DDS/lib AND EXISTS ${SIMPATH}/DDS/include)
    set(DDS_FOUND TRUE)
    set(DDS_PATH ${SIMPATH}/DDS)
    message(STATUS "DDS found at ${DDS_PATH}")
  else()
    set(DDS_FOUND FALSE)
    message(STATUS "DDS not found")
  endif()
endif(DDS_PATH)

if(DDS_FOUND)
  add_definitions(-DENABLE_DDS)
  set(DDS_INCLUDE_DIR ${DDS_PATH}/include)
  set(DDS_LIBRARY_DIR ${DDS_PATH}/lib)
endif(DDS_FOUND)
