# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test(sampa-header
            SOURCES testSampaHeader.cxx
            COMPONENT_NAME mchraw
            LABELS "muon;mch;raw"
            PUBLIC_LINK_LIBRARIES O2::MCHRawCommon Boost::boost)

o2_add_test(sampa-cluster
            SOURCES testSampaCluster.cxx
            COMPONENT_NAME mchraw
            LABELS "muon;mch;raw"
            PUBLIC_LINK_LIBRARIES O2::MCHRawCommon Boost::boost)

o2_add_test(sampa-bx-count
            SOURCES testSampaBunchCrossingCounter.cxx
            COMPONENT_NAME mchraw
            LABELS "muon;mch;raw"
            PUBLIC_LINK_LIBRARIES O2::MCHRawCommon Boost::boost)

if(benchmark_FOUND)
  o2_add_executable(sampa-header
                    COMPONENT_NAME mch
                    SOURCES benchSampaHeader.cxx
                    PUBLIC_LINK_LIBRARIES O2::MCHRawCommon benchmark::benchmark
                    IS_BENCHMARK)
endif()
