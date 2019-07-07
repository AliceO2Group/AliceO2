# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# FIXME: the LinkDef should not be in the public area

o2_add_library(Mergers
               SOURCES src/Merger.cxx src/MergerInfrastructureBuilder.cxx
                       src/MergerCache.cxx src/MergerBuilder.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework)

o2_target_root_dictionary(
  Mergers
  HEADERS include/Mergers/Merger.h
          include/Mergers/MergerConfig.h
          include/Mergers/MergeInterface.h
          include/Mergers/MergeInterfaceOverrideExample.h
          include/Mergers/MergerInfrastructureBuilder.h
          include/Mergers/MergerBuilder.h
          include/Mergers/MergerCache.h
  LINKDEF include/Mergers/LinkDef.h)

o2_add_executable(topology-example
                  SOURCES src/mergersTopologyExample.cxx
                  COMPONENT_NAME mergers
                  PUBLIC_LINK_LIBRARIES O2::Mergers)

o2_add_executable(benchmark-topology
                  SOURCES src/mergersBenchmarkTopology.cxx
                  COMPONENT_NAME mergers
                  PUBLIC_LINK_LIBRARIES O2::Mergers)

o2_add_test(InfrastructureBuilder
            SOURCES test/test_InfrastructureBuilder.cxx
            COMPONENT_NAME mergers
            PUBLIC_LINK_LIBRARIES O2::Mergers
            LABELS utils)
