# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(SimConfig
               SOURCES src/SimConfig.cxx src/ConfigurableParam.cxx
                       src/ConfigurableParamHelper.cxx src/SimCutParams.cxx
                       src/G4Params.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonUtils
                                     O2::DetectorsCommonDataFormats
                                     FairRoot::Base Boost::program_options
                                     Boost::filesystem)

o2_target_root_dictionary(SimConfig
                          HEADERS include/SimConfig/SimConfig.h
                                  include/SimConfig/SimCutParams.h
                                  include/SimConfig/ConfigurableParam.h
                                  include/SimConfig/ConfigurableParamHelper.h
                                  include/SimConfig/G4Params.h)

o2_add_test(Config
            SOURCES test/TestConfig.cxx
            COMPONENT_NAME SimConfig
            PUBLIC_LINK_LIBRARIES O2::SimConfig
            NO_BOOST_TEST)

# FIXME: not working ?
#
# * o2_add_test(ConfigurableParam
# * SOURCES test/TestConfigurableParam.cxx
# * COMPONENT_NAME SimConfig
# * PUBLIC_LINK_LIBRARIES O2::SimConfig
# * NO_BOOST_TEST)
