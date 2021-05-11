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
               SOURCES src/SimConfig.cxx 
                       src/SimCutParams.cxx		       
                       src/SimUserDecay.cxx		       
                       src/DigiParams.cxx src/G4Params.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonUtils
                                     O2::DetectorsCommonDataFormats
                                     FairRoot::Base Boost::program_options)


o2_target_root_dictionary(SimConfig
                          HEADERS include/SimConfig/SimConfig.h
                                  include/SimConfig/SimCutParams.h
                                  include/SimConfig/SimUserDecay.h
				  include/SimConfig/DigiParams.h
                                  include/SimConfig/G4Params.h)

o2_add_test(Config
            SOURCES test/TestConfig.cxx
            COMPONENT_NAME SimConfig
            PUBLIC_LINK_LIBRARIES O2::SimConfig
            NO_BOOST_TEST)

o2_add_test(simCutParam
            SOURCES test/testSimCutParam.cxx
            COMPONENT_NAME SimConfig
            PUBLIC_LINK_LIBRARIES O2::SimConfig)

# FIXME: not working ?
#
# * o2_add_test(ConfigurableParam
# * SOURCES test/TestConfigurableParam.cxx
# * COMPONENT_NAME SimConfig
# * PUBLIC_LINK_LIBRARIES O2::SimConfig
# * NO_BOOST_TEST)
