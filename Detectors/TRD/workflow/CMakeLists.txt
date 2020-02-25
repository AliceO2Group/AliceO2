# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

add_compile_options(-O0 -g -fPIC) 

o2_add_library(TRDWorkflow
               TARGETVARNAME targetName
               SOURCES src/TRDDigitizerSpec.cxx src/TRDDigitWriterSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::Steer O2::Algorithm O2::DataFormatsTRD O2::TRDSimulation O2::DetectorsBase O2::SimulationDataFormat O2::TRDBase)

           o2_target_root_dictionary(TRDWorkflow
               HEADERS include/TRDWorkflow/TRDDigitizerSpec.h)
