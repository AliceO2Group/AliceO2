# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsPHOS
               SOURCES src/PHOSBlockHeader.cxx
                       src/Cell.cxx
                       src/Digit.cxx
                       src/Cluster.cxx
                       src/MCLabel.cxx
                       src/TriggerRecord.cxx
                       src/CTF.cxx
                       src/BadChannelsMap.cxx
                       src/CalibParams.cxx
                       src/Pedestals.cxx
                       src/TriggerMap.cxx
                PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                                     O2::Headers
                                     O2::CCDB
                                     O2::MathUtils
                                     O2::DetectorsBase
                                     O2::PHOSBase
                                     O2::SimulationDataFormat
                                     Boost::serialization)

o2_target_root_dictionary(DataFormatsPHOS
                          HEADERS include/DataFormatsPHOS/PHOSBlockHeader.h
                                  include/DataFormatsPHOS/Cell.h
                                  include/DataFormatsPHOS/Digit.h
                                  include/DataFormatsPHOS/Cluster.h
                                  include/DataFormatsPHOS/MCLabel.h
                                  include/DataFormatsPHOS/TriggerRecord.h
                                  include/DataFormatsPHOS/CTF.h
                                  include/DataFormatsPHOS/BadChannelsMap.h
                                  include/DataFormatsPHOS/CalibParams.h
                                  include/DataFormatsPHOS/Pedestals.h
                                  include/DataFormatsPHOS/TriggerMap.h)
o2_add_test(Cell
            SOURCES test/testCell.cxx
            COMPONENT_NAME DataFormats-PHOS
            PUBLIC_LINK_LIBRARIES O2::DataFormatsPHOS
            LABELS phos dataformats)
