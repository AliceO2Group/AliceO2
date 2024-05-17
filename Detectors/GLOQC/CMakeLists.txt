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
# add_compile_options(-O0 -g -fPIC)

o2_add_library(GLOQC
               TARGETVARNAME targetName
               SOURCES src/MatchITSTPCQC.cxx
                       src/ITSTPCMatchingQCParams.cxx

  PUBLIC_LINK_LIBRARIES O2::Framework
                        O2::DataFormatsTPC
                        O2::DataFormatsITS
                        O2::SimConfig
                        O2::DataFormatsGlobalTracking
                        O2::MathUtils
                        O2::SimulationDataFormat
                        O2::ReconstructionDataFormats
                        O2::Steer
                        O2::GlobalTracking
                        O2::DCAFitter
                        O2::DetectorsVertexing)

o2_target_root_dictionary(GLOQC
                          HEADERS include/GLOQC/MatchITSTPCQC.h
                                  include/GLOQC/ITSTPCMatchingQCParams.h

)


