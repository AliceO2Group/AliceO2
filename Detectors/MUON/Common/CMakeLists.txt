# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(subsystems "mch;mid")

foreach(det IN LISTS subsystems)
  string(TOUPPER ${det} DET)
  o2_add_executable(dcs-ccdb
                    COMPONENT_NAME ${det}
                    SOURCES src/dcs-ccdb.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsDCS O2::CCDB O2::${DET}Conditions
                    TARGETVARNAME exe-dcs-ccdb)
  target_compile_definitions(${exe-dcs-ccdb} PRIVATE "MUON_SUBSYSTEM_${DET}")

  o2_add_executable(${det}-dcs-processor-workflow
                    COMPONENT_NAME calibration
                    SOURCES src/dcs-processor-workflow.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsDCS O2::${DET}Conditions
                    TARGETVARNAME exe-dcs-processor-workflow)
  target_compile_definitions(${exe-dcs-processor-workflow} PRIVATE "MUON_SUBSYSTEM_${DET}")

  o2_add_executable(${det}-dcs-sim-workflow
                    COMPONENT_NAME calibration
                    SOURCES src/${det}-dcs-sim-workflow.cxx
                    PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow O2::${DET}Conditions)
endforeach()

