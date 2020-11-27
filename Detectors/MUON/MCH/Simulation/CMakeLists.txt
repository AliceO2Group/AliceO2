# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MCHSimulation
               SOURCES src/Detector.cxx
                       src/Digitizer.cxx
                       src/Hit.cxx
                       src/Stepper.cxx
                       src/Stepper.h
                       src/Response.cxx
		       PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
		                             O2::MCHBase O2::DetectorsPassive
                                     O2::MCHMappingImpl4 O2::DetectorsBase
                                     O2::MCHGeometryCreator)

o2_target_root_dictionary(MCHSimulation
                          HEADERS include/MCHSimulation/Detector.h
                                  include/MCHSimulation/Digitizer.h
                                  include/MCHSimulation/Hit.h
                                  include/MCHSimulation/Response.h)

o2_data_file(COPY data DESTINATION Detectors/MCH/simulation)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
