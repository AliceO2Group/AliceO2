# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

foreach(impl RANGE 3 4)
        o2_add_executable(mapping-cli${impl}
                SOURCES cli.cxx
                COMPONENT_NAME mch
                PUBLIC_LINK_LIBRARIES O2::MCHMappingImpl${impl} Boost::program_options)
endforeach()

foreach(impl RANGE 3 4)
        o2_add_executable(mapping-export-to-tree${impl}
                SOURCES export-to-tree.cxx
                COMPONENT_NAME mch
                PUBLIC_LINK_LIBRARIES O2::MCHMappingImpl${impl} Boost::program_options
                ROOT::Tree
                TARGETVARNAME targetName)
        if(impl EQUAL 3)
            target_compile_definitions(${targetName} PRIVATE RUN2)
        endif()
endforeach()

