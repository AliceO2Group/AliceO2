# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(
  MathUtils
  SOURCES src/CachingTF1.cxx
          src/Cartesian.cxx
          src/Chebyshev3D.cxx
          src/Chebyshev3DCalc.cxx
  PUBLIC_LINK_LIBRARIES
    ROOT::Hist
    FairLogger::FairLogger
    O2::CommonConstants
    O2::GPUCommon
    ROOT::GenVector
    ROOT::Geom
    Vc::Vc)

o2_target_root_dictionary(
  MathUtils
  HEADERS include/MathUtils/Utils.h
          include/MathUtils/Chebyshev3D.h
          include/MathUtils/Chebyshev3DCalc.h
          include/MathUtils/fit.h
          include/MathUtils/Cartesian.h
          include/MathUtils/CartesianGPU.h
          include/MathUtils/CachingTF1.h
          include/MathUtils/RandomRing.h
          include/MathUtils/Primitive2D.h
          include/MathUtils/SMatrixGPU.h)

o2_add_test(
  CachingTF1
  SOURCES test/testCachingTF1.cxx
  COMPONENT_NAME MathUtils
  PUBLIC_LINK_LIBRARIES O2::MathUtils
  LABELS utils)

o2_add_test(
  Cartesian3D
  SOURCES test/testCartesian.cxx
  COMPONENT_NAME MathUtils
  PUBLIC_LINK_LIBRARIES O2::MathUtils
  LABELS utils)

o2_add_test(
  Utils
  SOURCES test/testUtils.cxx
  COMPONENT_NAME MathUtils
  PUBLIC_LINK_LIBRARIES O2::MathUtils
  LABELS utils)
