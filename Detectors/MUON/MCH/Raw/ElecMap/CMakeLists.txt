# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHRawElecMap
        SOURCES
        src/CH5L.cxx
        src/CH5R.cxx
        src/CH6L.cxx
        src/CH6R.cxx
        src/CH7L.cxx
        src/CH7R.cxx
        src/CH8L.cxx
        src/CH8R.cxx
        src/CH9L.cxx
        src/CH9R.cxx
        src/CH10L.cxx
        src/CH10R.cxx
        src/DsDetId.cxx
        src/DsElecId.cxx
        src/ElectronicMapperDummy.cxx
        src/ElectronicMapperGenerated.cxx
        src/ElectronicMapperString.cxx
        src/FeeLinkId.cxx
        src/MapCRU.cxx
        src/MapFEC.cxx
        src/Mapper.cxx
        src/dslist.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHRawImplHelpers Microsoft.GSL::GSL)


o2_add_executable(elecmap-cli
        SOURCES src/cli.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap Boost::program_options)

if(BUILD_TESTING)
        o2_add_test(elecmap
                SOURCES src/testElectronicMapper.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap)
        o2_add_test(dselecid
                SOURCES src/testDsElecId.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap)
        o2_add_test(feelinkid
                SOURCES src/testFeeLinkId.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap)
        o2_add_test(elecmap-string
                SOURCES src/testElectronicMapperString.cxx
                COMPONENT_NAME mchraw
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap)
endif()
