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
  NDRegression
  SOURCES src/NDRegression.cxx
  PUBLIC_LINK_LIBRARIES
    O2::MathUtils
    O2::Headers
    O2::CommonUtils
    ROOT::Minuit
    # ROOT::Hist
    # FairRoot::Base
    # O2::CommonConstants
    # O2::GPUCommon
    # ROOT::GenVector
    # ROOT::Geom
    # Vc::Vc
    )

o2_target_root_dictionary(
  NDRegression
  HEADERS include/NDRegression/NDRegression.h
          )

o2_add_test(
  NDRegression2DIdeal
  SOURCES test/NDRegression2DIdeal.cxx
  COMPONENT_NAME NDRegression
  PUBLIC_LINK_LIBRARIES O2::NDRegression
  LABELS init tfile ttree histo makefit)
o2_add_test(
  NDRegression2DGaus
  SOURCES test/NDRegression2DGaus.cxx
  COMPONENT_NAME NDRegression
  PUBLIC_LINK_LIBRARIES O2::NDRegression
  LABELS init tfile ttree histo makefit)
