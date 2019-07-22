# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(AliRoot_FOUND FALSE)

if(ALIROOT)

  # Check if AliRoot is really installed there
  if(EXISTS ${ALIROOT}/bin/aliroot
     AND EXISTS ${ALIROOT}/lib
     AND EXISTS ${ALIROOT}/include)

    # TODO this is really not the way it should be done
    include_directories(${ALIROOT}/include ${ALIROOT}/include/pythia)
    # TODO neither is this
    link_directories(${ALIROOT}/lib)

    set(AliRoot_FOUND TRUE)

    message(STATUS "AliRoot ... - found ${ALIROOT}")

  else()

    message(STATUS "AliRoot ... - not found")

  endif()
endif(ALIROOT)

if(NOT AliRoot_FOUND)
  if(AliRoot_FIND_REQUIRED)
    message(
      FATAL_ERROR
        "Please point to the AliRoot Core installation using -DALIROOT=<ALIROOT_CORE_INSTALL_DIR>"
      )
  endif(AliRoot_FIND_REQUIRED)
endif(NOT AliRoot_FOUND)
