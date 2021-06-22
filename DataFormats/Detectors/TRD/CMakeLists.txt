# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsTRD
               SOURCES src/TriggerRecord.cxx
                       src/LinkRecord.cxx
                       src/AngularResidHistos.cxx
                       src/Tracklet64.cxx
                       src/RawData.cxx
                       src/CompressedDigit.cxx
                       src/CTF.cxx
                       src/Digit.cxx
                       src/EventRecord.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonDataFormat O2::SimulationDataFormat)

o2_target_root_dictionary(DataFormatsTRD
               HEADERS include/DataFormatsTRD/TriggerRecord.h
                       include/DataFormatsTRD/TrackTriggerRecord.h
                       include/DataFormatsTRD/LinkRecord.h
                       include/DataFormatsTRD/Tracklet64.h
                       include/DataFormatsTRD/RawData.h
                       include/DataFormatsTRD/Constants.h
                       include/DataFormatsTRD/CalibratedTracklet.h
                       include/DataFormatsTRD/AngularResidHistos.h
                       include/DataFormatsTRD/HelperMethods.h
                       include/DataFormatsTRD/Hit.h
                       include/DataFormatsTRD/Digit.h
                       include/DataFormatsTRD/CTF.h
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
