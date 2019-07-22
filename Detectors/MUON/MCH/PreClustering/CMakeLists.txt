# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHPreClustering
               SOURCES src/PreClusterFinder.cxx
               PUBLIC_LINK_LIBRARIES O2::MCHBase O2::Framework)

o2_add_executable(preclusterizer-workflow
                  COMPONENT_NAME mch
                  SOURCES src/PreClusterizerWorkflow.cxx
                          src/DigitSamplerSpec.cxx src/PreClusterFinderSpec.cxx
                          src/PreClusterSinkSpec.cxx
                  PUBLIC_LINK_LIBRARIES O2::MCHPreClustering)
