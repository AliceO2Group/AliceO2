# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DetectorsPassive
               SOURCES src/Absorber.cxx
                       src/Cave.cxx
                       src/Dipole.cxx
                       src/Compensator.cxx
                       src/Pipe.cxx
                       src/Magnet.cxx
                       src/PassiveContFact.cxx
                       src/FrameStructure.cxx
                       src/Shil.cxx
                       src/Hall.cxx
                       src/HallSimParam.cxx
               PUBLIC_LINK_LIBRARIES O2::Field O2::DetectorsBase O2::SimConfig)

o2_target_root_dictionary(DetectorsPassive
                          HEADERS include/DetectorsPassive/Absorber.h
                                  include/DetectorsPassive/Cave.h
                                  include/DetectorsPassive/Dipole.h
                                  include/DetectorsPassive/Compensator.h
                                  include/DetectorsPassive/Magnet.h
                                  include/DetectorsPassive/PassiveContFact.h
                                  include/DetectorsPassive/Pipe.h
                                  include/DetectorsPassive/FrameStructure.h
                                  include/DetectorsPassive/Shil.h
                                  include/DetectorsPassive/Hall.h
                                  include/DetectorsPassive/HallSimParam.h
                          LINKDEF src/PassiveLinkDef.h)

# FIXME: if PutFrameInTop really depends on TRD, then the following can not work
# as TRD is built afterwards. So we have a dependency problem here potentially.
# o2_add_test_root_macro(FILENAME "macro/PutFrameInTop.C" PUBLIC_LINK_LIBRARIES
# DetectorsPassive FairRoot::Base TRDSimulation)
