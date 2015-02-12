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

set(AliRoot_FOUND FALSE)

if(ALIROOT)

  # Check if AliRoot is really installed there
  if(EXISTS ${ALIROOT}/bin/aliroot AND EXISTS ${ALIROOT}/lib AND EXISTS ${ALIROOT}/include)

    include_directories(
      ${ALIROOT}/include
      ${ALIROOT}/include/pythia
    )

    link_directories(${ALIROOT}/lib)

    set(AliRoot_FOUND TRUE)

    message(STATUS "AliRoot ... - found ${ALIROOT}")

  endif()
endif(ALIROOT)

if(NOT AliRoot_FOUND)
  if(AliRoot_FIND_REQUIRED)
    message(FATAL_ERROR "Please point to the AliRoot Core installation using -DALIROOT=<ALIROOT_CORE_INSTALL_DIR>")
  endif(AliRoot_FIND_REQUIRED)
endif(NOT AliRoot_FOUND)
