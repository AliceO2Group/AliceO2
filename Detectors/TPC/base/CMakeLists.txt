# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TPCBase
               SOURCES src/CalArray.cxx
                       src/CalDet.cxx
                       src/CDBInterface.cxx
                       src/ContainerFactory.cxx
                       src/CRU.cxx
                       src/Digit.cxx
                       src/DigitPos.cxx
                       src/ModelGEM.cxx
                       src/FECInfo.cxx
                       src/Mapper.cxx
                       src/PadInfo.cxx
                       src/PadPos.cxx
                       src/PadRegionInfo.cxx
                       src/PadROCPos.cxx
                       src/PadSecPos.cxx
                       src/Painter.cxx
                       src/ParameterDetector.cxx
                       src/ParameterElectronics.cxx
                       src/ParameterGas.cxx
                       src/ParameterGEM.cxx
                       src/PartitionInfo.cxx
                       src/RandomRing.cxx
                       src/ROC.cxx
                       src/Sector.cxx
               PUBLIC_LINK_LIBRARIES Vc::Vc Boost::boost O2::DataFormatsTPC
                                     O2::CCDB FairRoot::Base)

o2_target_root_dictionary(TPCBase
                          HEADERS include/TPCBase/CalArray.h
                                  include/TPCBase/CalDet.h
                                  include/TPCBase/CDBInterface.h
                                  include/TPCBase/ContainerFactory.h
                                  include/TPCBase/CRU.h
                                  include/TPCBase/Digit.h
                                  include/TPCBase/DigitPos.h
                                  include/TPCBase/ModelGEM.h
                                  include/TPCBase/FECInfo.h
                                  include/TPCBase/Mapper.h
                                  include/TPCBase/PadInfo.h
                                  include/TPCBase/PadPos.h
                                  include/TPCBase/PadRegionInfo.h
                                  include/TPCBase/PadROCPos.h
                                  include/TPCBase/PadSecPos.h
                                  include/TPCBase/Painter.h
                                  include/TPCBase/ParameterDetector.h
                                  include/TPCBase/ParameterElectronics.h
                                  include/TPCBase/ParameterGas.h
                                  include/TPCBase/ParameterGEM.h
                                  include/TPCBase/PartitionInfo.h
                                  include/TPCBase/RandomRing.h
                                  include/TPCBase/ROC.h
                                  include/TPCBase/Sector.h)

o2_add_test(Base
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCBase
            SOURCES test/testTPCBase.cxx
            LABELS tpc)

o2_add_test(CalDet
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCBase
            SOURCES test/testTPCCalDet.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS tpc)

o2_add_test(Mapper
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCBase
            SOURCES test/testTPCMapper.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS tpc)

o2_add_test(Parameters
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCBase
            SOURCES test/testTPCParameters.cxx
            LABELS tpc)

o2_add_test(CCDBInterface
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCBase
            SOURCES test/testTPCCDBInterface.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS tpc)

o2_data_file(COPY files DESTINATION Detectors/TPC)
