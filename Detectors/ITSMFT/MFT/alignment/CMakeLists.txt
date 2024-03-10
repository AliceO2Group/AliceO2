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

o2_add_library(MFTAlignment
        SOURCES src/AlignConfig.cxx
                src/Aligner.cxx
                src/AlignPointControl.cxx
                src/AlignPointHelper.cxx
                src/AlignSensorHelper.cxx
                src/RecordsToAlignParams.cxx
                src/TracksToRecords.cxx
        PUBLIC_LINK_LIBRARIES O2::MFTBase
                O2::DataFormatsMFT
                O2::MFTTracking
                O2::CCDB
                O2::Steer
                O2::ForwardAlign
                ROOT::TreePlayer)

o2_target_root_dictionary(MFTAlignment
        HEADERS include/MFTAlignment/AlignConfig.h
                include/MFTAlignment/Aligner.h
                include/MFTAlignment/AlignPointControl.h
                include/MFTAlignment/AlignPointHelper.h
                include/MFTAlignment/AlignSensorHelper.h
                include/MFTAlignment/RecordsToAlignParams.h
                include/MFTAlignment/TracksToRecords.h
        LINKDEF src/MFTAlignmentLinkDef.h)
