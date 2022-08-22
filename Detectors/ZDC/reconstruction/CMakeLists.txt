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

o2_add_library(ZDCReconstruction
               SOURCES src/CTFCoder.cxx
                       src/CTFHelper.cxx
                       src/DigiReco.cxx
                       src/RecoParamZDC.cxx
                       src/ZDCTDCParam.cxx
                       src/ZDCEnergyParam.cxx
                       src/ZDCTowerParam.cxx
                       src/BaselineParam.cxx
                       src/RecoConfigZDC.cxx
                       src/DigiRecoTest.cxx
                       src/ZDCTDCCorr.cxx
              PUBLIC_LINK_LIBRARIES O2::ZDCBase
                                     O2::DataFormatsZDC
                                     O2::ZDCSimulation
                                     O2::DetectorsRaw
                                     AliceO2::InfoLogger
                                     O2::rANS
                                     Microsoft.GSL::GSL)

o2_target_root_dictionary(ZDCReconstruction
                          HEADERS include/ZDCReconstruction/RecoConfigZDC.h
                                  include/ZDCReconstruction/RecoParamZDC.h
                                  include/ZDCReconstruction/ZDCTDCParam.h
                                  include/ZDCReconstruction/ZDCEnergyParam.h
                                  include/ZDCReconstruction/ZDCTowerParam.h
                                  include/ZDCReconstruction/BaselineParam.h
                                  include/ZDCReconstruction/ZDCTDCCorr.h)
