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

o2_add_library(GlobalTracking
               TARGETVARNAME targetName
               SOURCES src/MatchTPCITS.cxx
                       src/MatchTOF.cxx
                       src/MatchTPCITSParams.cxx
                       src/MatchCosmics.cxx
                       src/MatchCosmicsParams.cxx
                       src/MatchGlobalFwd.cxx
                       src/MatchGlobalFwdAssessment.cxx
                       src/MatchITSTPCQC.cxx
                       src/ITSTPCMatchingQCParams.cxx
                       src/MatchGlobalFwdParam.cxx
  PUBLIC_LINK_LIBRARIES O2::Framework
                        O2::DataFormatsTPC
                        O2::DataFormatsITSMFT
                        O2::DataFormatsITS
                        O2::DataFormatsFT0
                        O2::DataFormatsTOF
                        O2::DataFormatsTRD
                        O2::ITSReconstruction
                        O2::FT0Reconstruction
                        O2::TPCFastTransformation
                        O2::GPUO2Interface
                        O2::TPCBase
                        O2::TPCReconstruction
                        O2::TPCCalibration
                        O2::TOFBase
                        O2::TOFCalibration
                        O2::TOFWorkflowUtils
                        O2::SimConfig
                        O2::DataFormatsFT0
                        O2::DataFormatsGlobalTracking
                        O2::ITStracking
                        O2::MFTTracking
                        O2::MCHTracking
                        O2::MathUtils
                        O2::ReconstructionDataFormats
                        O2::Steer)

o2_target_root_dictionary(GlobalTracking
                          HEADERS include/GlobalTracking/MatchTPCITSParams.h
                                  include/GlobalTracking/MatchGlobalFwd.h
                                  include/GlobalTracking/MatchGlobalFwdParam.h
                                  include/GlobalTracking/MatchGlobalFwdAssessment.h
                                  include/GlobalTracking/MatchTOF.h
                                  include/GlobalTracking/MatchCosmics.h
                                  include/GlobalTracking/MatchCosmicsParams.h
                                  include/GlobalTracking/MatchITSTPCQC.h
                                  include/GlobalTracking/ITSTPCMatchingQCParams.h
                                  include/GlobalTracking/TrackMethods.h
                                  include/GlobalTracking/TrackCuts.h)


if (OpenMP_CXX_FOUND)
  target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
