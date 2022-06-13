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

o2_add_library(DataFormatsZDC
               SOURCES src/ChannelData.cxx src/BCData.cxx src/BCRecData.cxx src/RecEvent.cxx src/RecEventAux.cxx src/RawEventData.cxx
                       src/OrbitRawData.cxx src/OrbitRecData.cxx src/OrbitData.cxx src/ZDCTDCData.cxx src/ZDCEnergy.cxx src/ZDCWaveform.cxx
                       src/CTF.cxx src/RecEventFlat.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonConstants O2::CommonDataFormat O2::DetectorsCalibration
                   O2::ZDCBase ROOT::MathCore FairRoot::Base
                   O2::SimulationDataFormat
                   O2::MathUtils Microsoft.GSL::GSL)

o2_target_root_dictionary(DataFormatsZDC
               HEADERS include/DataFormatsZDC/Hit.h include/DataFormatsZDC/MCLabel.h
                       include/DataFormatsZDC/BCData.h include/DataFormatsZDC/ChannelData.h
                       include/DataFormatsZDC/OrbitData.h include/DataFormatsZDC/CTF.h
                       include/DataFormatsZDC/RecEvent.h include/DataFormatsZDC/RecEventAux.h include/DataFormatsZDC/RecEventFlat.h
                       include/DataFormatsZDC/OrbitRawData.h include/DataFormatsZDC/ZDCTDCData.h
                       include/DataFormatsZDC/BCRecData.h include/DataFormatsZDC/ZDCEnergy.h include/DataFormatsZDC/ZDCWaveform.h
                       include/DataFormatsZDC/OrbitRecData.h include/DataFormatsZDC/RawEventData.h)
