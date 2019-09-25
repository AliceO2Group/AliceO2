# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(GlobalTracking
               SOURCES src/MatchTPCITS.cxx src/MatchTOF.cxx src/CalibTOF.cxx
                       src/CollectCalibInfoTOF.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsITS
                                     O2::DataFormatsFT0
                                     O2::DataFormatsTOF
                                     O2::TPCFastTransformation
                                     O2::GPUTracking
                                     O2::TPCBase
                                     O2::TPCReconstruction
                                     O2::TOFBase)

o2_target_root_dictionary(GlobalTracking
                          HEADERS include/GlobalTracking/MatchTPCITS.h
                                  include/GlobalTracking/MatchTOF.h
                                  include/GlobalTracking/CalibTOF.h
                                  include/GlobalTracking/CollectCalibInfoTOF.h)
