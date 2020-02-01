# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TRDSimulation
               TARGETVARNAME targetName
               SOURCES src/Detector.cxx
                       src/TRsim.cxx
                       src/Digitizer.cxx
                       src/TRDSimParams.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase O2::SimulationDataFormat O2::TRDBase)

o2_target_root_dictionary(TRDSimulation
                          HEADERS include/TRDSimulation/Detector.h
                                  include/TRDSimulation/TRsim.h
                                  include/TRDSimulation/Digitizer.h
                                  include/TRDSimulation/TRDSimParams.h)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

o2_data_file(COPY data DESTINATION Detectors/TRD/simulation)
