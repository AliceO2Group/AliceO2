# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MFTBase
               SOURCES src/GeometryTGeo.cxx
                       src/Geometry.cxx
                       src/GeometryBuilder.cxx
                       src/VSegmentation.cxx
                       src/Segmentation.cxx
                       src/HalfSegmentation.cxx
                       src/HalfDiskSegmentation.cxx
                       src/LadderSegmentation.cxx
                       src/ChipSegmentation.cxx
                       src/HalfDetector.cxx
                       src/HalfDisk.cxx
                       src/Ladder.cxx
                       src/Flex.cxx
                       src/Support.cxx
                       src/HeatExchanger.cxx
                       src/HalfCone.cxx
                       src/PowerSupplyUnit.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase O2::ITSMFTBase
                                     O2::ITSMFTSimulation ROOT::XMLIO)

o2_target_root_dictionary(MFTBase
                          HEADERS include/MFTBase/Constants.h
                                  include/MFTBase/GeometryTGeo.h
                                  include/MFTBase/Geometry.h
                                  include/MFTBase/GeometryBuilder.h
                                  include/MFTBase/VSegmentation.h
                                  include/MFTBase/Segmentation.h
                                  include/MFTBase/HalfSegmentation.h
                                  include/MFTBase/HalfDiskSegmentation.h
                                  include/MFTBase/LadderSegmentation.h
                                  include/MFTBase/ChipSegmentation.h
                                  include/MFTBase/HalfDetector.h
                                  include/MFTBase/HalfDisk.h
                                  include/MFTBase/Ladder.h
                                  include/MFTBase/Flex.h
                                  include/MFTBase/Support.h
                                  include/MFTBase/HeatExchanger.h
                                  include/MFTBase/HalfCone.h
                                  include/MFTBase/PowerSupplyUnit.h)
