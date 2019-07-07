# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MIDSimulation
               SOURCES src/ChamberEfficiencyResponse.cxx
                       src/ChamberHV.cxx
                       src/ChamberResponse.cxx
                       src/ChamberResponseParams.cxx
                       src/ClusterLabeler.cxx
                       src/Detector.cxx
                       src/Digitizer.cxx
                       src/DigitsMerger.cxx
                       src/DigitsPacker.cxx
                       src/Geometry.cxx
                       src/Hit.cxx
                       src/Materials.cxx
                       src/MCClusterLabel.cxx
                       src/MCLabel.cxx
                       src/PreClusterLabeler.cxx
                       src/Stepper.cxx
                       src/TrackLabeler.cxx
               PUBLIC_LINK_LIBRARIES O2::MIDBase O2::CommonDataFormat
                                     O2::DetectorsBase O2::SimulationDataFormat
                                     O2::MIDClustering)

o2_target_root_dictionary(
  MIDSimulation
  HEADERS include/MIDSimulation/ChamberEfficiencyResponse.h
          include/MIDSimulation/ChamberHV.h
          include/MIDSimulation/ChamberResponse.h
          include/MIDSimulation/ChamberResponseParams.h
          include/MIDSimulation/ClusterLabeler.h
          include/MIDSimulation/ColumnDataMC.h
          include/MIDSimulation/Detector.h
          include/MIDSimulation/Digitizer.h
          include/MIDSimulation/DigitsMerger.h
          include/MIDSimulation/DigitsPacker.h
          include/MIDSimulation/Hit.h
          include/MIDSimulation/MCClusterLabel.h
          include/MIDSimulation/MCLabel.h
          include/MIDSimulation/Stepper.h)

if(BUILD_TESTING)
  add_subdirectory(test)

  o2_add_test_root_macro(macros/drawMIDGeometry.C
                         PUBLIC_LINK_LIBRARIES FairRoot::Base
                                               O2::DetectorsPassive
                                               O2::MIDSimulation
                         LABELS "muon;mid")
endif()
