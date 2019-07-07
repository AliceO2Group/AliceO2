# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(ITSSimulation
               SOURCES src/V11Geometry.cxx src/V1Layer.cxx src/V3Layer.cxx
                       src/Detector.cxx src/V3Services.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSBase O2::ITSMFTSimulation
                                     ROOT::Physics)

o2_target_root_dictionary(ITSSimulation
                          HEADERS include/ITSSimulation/Detector.h
                                  include/ITSSimulation/V1Layer.h
                                  include/ITSSimulation/V3Layer.h
                                  include/ITSSimulation/V11Geometry.h
                                  include/ITSSimulation/V3Services.h)
