# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test(
  DigitContainer
  LABELS tpc
  PUBLIC_LINK_LIBRARIES O2::TPCSimulation
  COMPONENT_NAME tpc
  SOURCES testTPCDigitContainer.cxx
  ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(
  ElectronTransport
  LABELS tpc
  PUBLIC_LINK_LIBRARIES O2::TPCSimulation
  COMPONENT_NAME tpc
  SOURCES testTPCElectronTransport.cxx)

o2_add_test(
  GEMAmplification
  LABELS tpc
         long
  PUBLIC_LINK_LIBRARIES O2::TPCSimulation
  COMPONENT_NAME tpc
  SOURCES testTPCGEMAmplification.cxx
  ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
  TIMEOUT 200)

o2_add_test(
  SAMPAProcessing
  LABELS tpc
  PUBLIC_LINK_LIBRARIES O2::TPCSimulation
  COMPONENT_NAME tpc
  SOURCES testTPCSAMPAProcessing.cxx
  ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(
  Simulation
  LABELS tpc
  PUBLIC_LINK_LIBRARIES O2::TPCSimulation
  COMPONENT_NAME tpc
  SOURCES testTPCSimulation.cxx)
