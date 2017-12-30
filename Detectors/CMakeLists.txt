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

# Libraries
add_subdirectory(Base)
add_subdirectory(Passive)
add_subdirectory(ITSMFT)
add_subdirectory(TPC)
add_subdirectory(EMCAL)
add_subdirectory(TOF)
add_subdirectory(TRD)
add_subdirectory(FIT)
add_subdirectory(PHOS)

Install(DIRECTORY Geometry gconfig DESTINATION share/Detectors/)
