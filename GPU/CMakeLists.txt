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

set(ALIGPU_BUILD_TYPE "O2")
#option(ENABLE_CUDA   "Build GPU tracker using CUDA"   OFF)
option(ENABLE_OPENCL "Build GPU tracker using OpenCL" OFF)
option(ENABLE_HIP    "Build GPU tracker using HIP"    OFF)

# Libraries
add_subdirectory(Common)
add_subdirectory(GPUTracking)
add_subdirectory(TPCFastTransformation)
add_subdirectory(TPCSpaceChargeBase)
