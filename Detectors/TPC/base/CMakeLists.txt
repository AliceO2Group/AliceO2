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

o2_add_library(TPCBase
               SOURCES src/CalArray.cxx
                       src/CalDet.cxx
                       src/CDBInterface.cxx
                       src/ContainerFactory.cxx
                       src/CRU.cxx
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
                       src/ROC.cxx
                       src/Sector.cxx
                       src/Utils.cxx
                       src/ZeroSuppress.cxx
                       src/CRUCalibHelpers.cxx
                       src/IonTailSettings.cxx
               PUBLIC_LINK_LIBRARIES Vc::Vc Boost::boost O2::DataFormatsTPC
                                     O2::DetectorsRaw O2::CCDB FairRoot::Base)

o2_target_root_dictionary(TPCBase
                          HEADERS include/TPCBase/CalArray.h
                                  include/TPCBase/CalDet.h
                                  include/TPCBase/CDBInterface.h
                                  include/TPCBase/ContainerFactory.h
                                  include/TPCBase/CRU.h
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
                                  include/TPCBase/ROC.h
                                  include/TPCBase/Sector.h
                                  include/TPCBase/ZeroSuppress.h
                                  include/TPCBase/Utils.h
                                  include/TPCBase/CRUCalibHelpers.h
                                  include/TPCBase/IonTailSettings.h)
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

o2_add_test(RDHUtils
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCBase
            SOURCES test/testRDHUtils.cxx
            LABELS tpc)

#o2_add_test(CCDBInterface
#            COMPONENT_NAME tpc
#            PUBLIC_LINK_LIBRARIES O2::TPCBase
#            SOURCES test/testTPCCDBInterface.cxx
#            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
#            LABELS tpc)

o2_data_file(COPY files DESTINATION Detectors/TPC)
