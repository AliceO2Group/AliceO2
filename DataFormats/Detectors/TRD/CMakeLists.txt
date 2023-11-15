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

o2_add_library(DataFormatsTRD
               SOURCES src/TriggerRecord.cxx
                       src/LinkRecord.cxx
                       src/AngularResidHistos.cxx
                       src/GainCalibHistos.cxx
                       src/T0FitHistos.cxx
                       src/DcsCcdbObjects.cxx
                       src/Tracklet64.cxx
                       src/RawData.cxx
                       src/HelperMethods.cxx
                       src/RawDataStats.cxx
                       src/CompressedDigit.cxx
                       src/CTF.cxx
                       src/Digit.cxx
                       src/KrCluster.cxx
                       src/NoiseCalibration.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonDataFormat O2::SimulationDataFormat)

o2_target_root_dictionary(DataFormatsTRD
               HEADERS include/DataFormatsTRD/TriggerRecord.h
                       include/DataFormatsTRD/TrackTriggerRecord.h
                       include/DataFormatsTRD/LinkRecord.h
                       include/DataFormatsTRD/Tracklet64.h
                       include/DataFormatsTRD/RawData.h
                       include/DataFormatsTRD/Constants.h
                       include/DataFormatsTRD/CalibratedTracklet.h
                       include/DataFormatsTRD/DcsCcdbObjects.h
                       include/DataFormatsTRD/AngularResidHistos.h
                       include/DataFormatsTRD/GainCalibHistos.h
                       include/DataFormatsTRD/T0FitHistos.h
                       include/DataFormatsTRD/HelperMethods.h
                       include/DataFormatsTRD/Hit.h
                       include/DataFormatsTRD/Digit.h
                       include/DataFormatsTRD/KrCluster.h
                       include/DataFormatsTRD/KrClusterTriggerRecord.h
                       include/DataFormatsTRD/NoiseCalibration.h
                       include/DataFormatsTRD/PHData.h
                       include/DataFormatsTRD/RawDataStats.h
                       include/DataFormatsTRD/CTF.h
                       include/DataFormatsTRD/CalVdriftExB.h
                       include/DataFormatsTRD/CalGain.h
                       include/DataFormatsTRD/CalT0.h
                       include/DataFormatsTRD/SignalArray.h
                       include/DataFormatsTRD/CompressedDigit.h
                       include/DataFormatsTRD/CompressedHeader.h)

o2_add_test(Digit
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
            SOURCES test/testDigit.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS trd
)

o2_add_test(RawData
            COMPONENT_NAME trd
            PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
            SOURCES test/testRawData.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS trd
            )
