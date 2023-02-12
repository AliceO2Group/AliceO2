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

o2_add_library(CommonUtils
               SOURCES src/TreeStream.cxx src/TreeStreamRedirector.cxx
                       src/RootChain.cxx src/CompStream.cxx src/ShmManager.cxx
                       src/ValueMonitor.cxx
                       src/StringUtils.cxx
                       src/ConfigurableParamHelper.cxx src/ConfigurableParam.cxx src/RootSerializableKeyValueStore.cxx
                       src/KeyValParam.cxx
                       src/FileSystemUtils.cxx
                       src/FIFO.cxx
                       src/FileFetcher.cxx
                       src/VerbosityConfig.cxx
                       src/BoostHistogramUtils.cxx
                       src/NameConf.cxx
                       src/IRFrameSelector.cxx
                       src/DebugStreamer.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Hist ROOT::Tree Boost::iostreams O2::CommonDataFormat O2::Headers
                                     FairLogger::FairLogger O2::MathUtils TBB::tbb)

o2_target_root_dictionary(CommonUtils
                          HEADERS include/CommonUtils/TreeStream.h
                                  include/CommonUtils/TreeStreamRedirector.h
                                  include/CommonUtils/RootChain.h
                                  include/CommonUtils/BoostHistogramUtils.h
                                  include/CommonUtils/BoostSerializer.h
                                  include/CommonUtils/ShmManager.h
                                  include/CommonUtils/RngHelper.h
                                  include/CommonUtils/StringUtils.h
                                  include/CommonUtils/ValueMonitor.h
                                  include/CommonUtils/MemFileHelper.h
                                  include/CommonUtils/ConfigurableParam.h
                                  include/CommonUtils/ConfigurableParamHelper.h
                                  include/CommonUtils/ConfigurationMacroHelper.h
                                  include/CommonUtils/RootSerializableKeyValueStore.h
                                  include/CommonUtils/KeyValParam.h
                                  include/CommonUtils/VerbosityConfig.h
                                  include/CommonUtils/FileFetcher.h
                                  include/CommonUtils/NameConf.h
                                  include/CommonUtils/IRFrameSelector.h
                                  include/CommonUtils/DebugStreamer.h)

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
            PUBLIC_LINK_LIBRARIES O2::CommonUtils)

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

o2_add_executable(treemergertool
            COMPONENT_NAME CommonUtils
          SOURCES src/TreeMergerTool.cxx
            PUBLIC_LINK_LIBRARIES O2::CommonUtils Boost::program_options ROOT::Core)
