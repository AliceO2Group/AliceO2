# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CommonUtils
               SOURCES src/TreeStream.cxx src/TreeStreamRedirector.cxx
                       src/RootChain.cxx src/CompStream.cxx src/ShmManager.cxx
	               src/ValueMonitor.cxx
                       src/ConfigurableParamHelper.cxx src/ConfigurableParam.cxx src/RootSerializableKeyValueStore.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Hist ROOT::Tree Boost::iostreams O2::CommonDataFormat O2::Headers
                                     FairLogger::FairLogger)

o2_target_root_dictionary(CommonUtils
                          HEADERS include/CommonUtils/TreeStream.h
                                  include/CommonUtils/TreeStreamRedirector.h
                                  include/CommonUtils/RootChain.h
                                  include/CommonUtils/BoostSerializer.h
                                  include/CommonUtils/ShmManager.h
                                  include/CommonUtils/RngHelper.h
                                  include/CommonUtils/StringUtils.h
                                  include/CommonUtils/ValueMonitor.h
				  include/CommonUtils/MemFileHelper.h
                                  include/CommonUtils/ConfigurableParam.h
                                  include/CommonUtils/ConfigurableParamHelper.h
                                  include/CommonUtils/ConfigurationMacroHelper.h
				  include/CommonUtils/RootSerializableKeyValueStore.h)

o2_add_test(TreeStream
            COMPONENT_NAME CommonUtils
            LABELS utils
            SOURCES test/testTreeStream.cxx
            PUBLIC_LINK_LIBRARIES O2::CommonUtils O2::ReconstructionDataFormats)

o2_add_test(BoostSerializer
            COMPONENT_NAME CommonUtils
            LABELS utils
            SOURCES test/testBoostSerializer.cxx
            PUBLIC_LINK_LIBRARIES O2::CommonUtils Boost::serialization)

o2_add_test(CompStream
            COMPONENT_NAME CommonUtils
            LABELS utils
            SOURCES test/testCompStream.cxx
            PUBLIC_LINK_LIBRARIES O2::CommonUtils Boost::filesystem)

o2_add_test(ValueMonitor
            COMPONENT_NAME CommonUtils
            LABELS utils
            SOURCES test/testValueMonitor.cxx
            PUBLIC_LINK_LIBRARIES O2::CommonUtils)

o2_add_test(PropertyMapIO
            COMPONENT_NAME CommonUtils
            LABELS utils
            SOURCES test/testRootSerializableKeyValueStore.cxx
            PUBLIC_LINK_LIBRARIES O2::CommonUtils)

o2_add_test(MemFileHelper
            COMPONENT_NAME CommonUtils
            LABELS utils
            SOURCES test/testMemFileHelper.cxx
            PUBLIC_LINK_LIBRARIES O2::CommonUtils)
