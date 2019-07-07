# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(flp2epn
               SOURCES src/O2FLPex.cxx src/O2EPNex.cxx
               PUBLIC_LINK_LIBRARIES O2::Device)

o2_add_executable(flp
                  SOURCES src/runFLP.cxx
                  PUBLIC_LINK_LIBRARIES O2::flp2epn
                  COMPONENT_NAME example)

o2_add_executable(epn
                  SOURCES src/runEPN.cxx
                  PUBLIC_LINK_LIBRARIES O2::flp2epn
                  COMPONENT_NAME example)

# FIXME: runProxy.cxx references no longer existing <devices/FairMQProxy.h> ?
# o2_add_executable(Proxy SOURCES src/runProxy.cxx PUBLIC_LINK_LIBRARIES
# O2::flp2epn COMPONENT_NAME example)
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/startFLP2EPN.sh.in
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/o2-start-flp2epn.sh)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/flp2epn.json
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/flp2epn.json)
