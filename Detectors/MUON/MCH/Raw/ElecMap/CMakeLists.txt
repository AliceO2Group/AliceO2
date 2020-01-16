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
        src/DsElecId.cxx
        src/CH5L.cxx
        src/CH5R.cxx
        src/CH6L.cxx
        src/CH6R.cxx
        src/CH7L.cxx
        src/CH7R.cxx
        src/CruLinkId.cxx
        src/DsDetId.cxx
        src/ElectronicMapperDummy.cxx
        src/ElectronicMapperGenerated.cxx
        src/dslist.cxx
        src/Mapper.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHRawImplHelpers ms_gsl::ms_gsl)

if(BUILD_TESTING)
        o2_add_test(elecmap-dummy
                SOURCES src/testElectronicMapper.cxx
                COMPONENT_NAME mch
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap)
        o2_add_test(dselecid
                SOURCES src/testDsElecId.cxx
                COMPONENT_NAME mch
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap)
        o2_add_test(crulinkid
                SOURCES src/testCruLinkId.cxx
                COMPONENT_NAME mch
                LABELS "muon;mch;raw"
                PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap)
endif()
