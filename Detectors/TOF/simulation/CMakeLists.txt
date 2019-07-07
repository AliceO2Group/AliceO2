# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TOFSimulation
               SOURCES src/Detector.cxx src/Digitizer.cxx src/DigitizerTask.cxx
                       src/Strip.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase O2::TOFBase
                                     O2::SimulationDataFormat)

o2_target_root_dictionary(TOFSimulation
                          HEADERS include/TOFSimulation/Detector.h
                                  include/TOFSimulation/Digitizer.h
                                  include/TOFSimulation/DigitizerTask.h
                                  include/TOFSimulation/Strip.h)
