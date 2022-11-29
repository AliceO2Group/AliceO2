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

add_compile_options(-O0 -g -fPIC)

o2_add_library(Align
               SOURCES  src/GeometricalConstraint.cxx
                        src/DOFSet.cxx
                        src/AlignableDetector.cxx
                        src/AlignableDetectorITS.cxx
                        #src/AlignableDetectorHMPID.cxx
                        src/AlignableDetectorTOF.cxx
                        #src/AlignableDetectorTPC.cxx
                        src/AlignableDetectorTRD.cxx
                        src/DOFStatistics.cxx
                        src/Millepede2Record.cxx
                        src/AlignmentPoint.cxx
                        src/ResidualsController.cxx
                        src/ResidualsControllerFast.cxx
                        src/AlignableSensor.cxx
                        #src/AlignableSensorHMPID.cxx
                        src/AlignableSensorITS.cxx
                        src/AlignableSensorTOF.cxx
                        #src/AlignableSensorTPC.cxx
                        src/AlignableSensorTRD.cxx
                        src/Controller.cxx
                        src/AlignmentTrack.cxx
                        src/AlignableVolume.cxx
                        src/EventVertex.cxx
                        src/AlignConfig.cxx
                        src/Mille.cxx
               PUBLIC_LINK_LIBRARIES O2::FrameworkLogger
                                     O2::Steer
                                     O2::ReconstructionDataFormats
                                     O2::DetectorsCommonDataFormats
                                     O2::DetectorsBase
                                     O2::ITSBase
                                     O2::ITStracking
                                     O2::TRDBase
                                     O2::TOFBase
                                     O2::DataFormatsGlobalTracking
                                     ROOT::Core
                                     ROOT::Geom
                                     ROOT::MathCore
                                     ROOT::Matrix
                                     ROOT::Hist
                                     ROOT::RIO
                                     ROOT::Tree)

o2_target_root_dictionary(
  Align
  HEADERS include/Align/DOFSet.h
          include/Align/AlignableDetector.h
          include/Align/AlignableDetectorITS.h
          include/Align/AlignableDetectorTOF.h
          #include/Align/AlignableDetectorTPC.h
          include/Align/AlignableDetectorTRD.h
          #include/Align/AlignableDetectorHMPID.h
          include/Align/Millepede2Record.h
          include/Align/AlignmentPoint.h
          include/Align/AlignableSensor.h
          include/Align/AlignableSensorITS.h
          include/Align/AlignableSensorTOF.h
          #include/Align/AlignableSensorTPC.h
          include/Align/AlignableSensorTRD.h
          #include/Align/AlignableSensorHMPID.h
          include/Align/Controller.h
          include/Align/AlignmentTrack.h
          include/Align/AlignableVolume.h
          include/Align/EventVertex.h
          include/Align/ResidualsController.h
          include/Align/ResidualsControllerFast.h
          include/Align/GeometricalConstraint.h
          include/Align/DOFStatistics.h
          include/Align/utils.h
          include/Align/AlignConfig.h
          )

add_subdirectory(Workflow)
add_subdirectory(macro)
