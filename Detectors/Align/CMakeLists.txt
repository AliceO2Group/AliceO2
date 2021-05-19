# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(
  Align
  SOURCES src/GeometricalConstraint.cxx
          src/AlignableDetector.cxx
          # src/AlignableDetectorHMPID.cxx src/AlignableDetectorITS.cxx src/AlignableDetectorTOF.cxx
          # src/AlignableDetectorTPC.cxx src/AlignableDetectorTRD.cxx
          src/DOFStatistics.cxx
          src/Millepede2Record.cxx
          src/AlignmentPoint.cxx
          src/ResidualsController.cxx
          src/ResidualsControllerFast.cxx
          src/AlignableSensor.cxx
          # src/AlignableSensorHMPID.cxx src/AlignableSensorITS.cxx src/AlignableSensorTOF.cxx
          # src/AlignableSensorTPC.cxx src/AlignableSensorTRD.cxx
          src/Controller.cxx
          src/AlignmentTrack.cxx
          src/AlignableVolume.cxx
          src/EventVertex.cxx
          src/Mille.cxx
  PUBLIC_LINK_LIBRARIES
    O2::FrameworkLogger
    O2::ReconstructionDataFormats
    O2::DetectorsCommonDataFormats
    O2::DetectorsBase
    ROOT::Core
    ROOT::Geom
    ROOT::Gpad
    ROOT::Graf3d
    ROOT::MathCore
    ROOT::Matrix
    ROOT::Hist
    ROOT::RIO
    ROOT::Tree)

o2_target_root_dictionary(
  Align
  HEADERS include/Align/AlignableDetector.h
          # include/Align/AlignableDetectorITS.h include/Align/AlignableDetectorTOF.h
          # include/Align/AlignableDetectorTPC.h include/Align/AlignableDetectorTRD.h
          # include/Align/AlignableDetectorHMPID.h
          include/Align/Millepede2Record.h
          include/Align/AlignmentPoint.h
          include/Align/AlignableSensor.h
          # include/Align/AlignableSensorITS.h include/Align/AlignableSensorTOF.h include/Align/AlignableSensorTPC.h
          # include/Align/AlignableSensorTRD.h include/Align/AlignableSensorHMPID.h
          include/Align/Controller.h
          include/Align/AlignmentTrack.h
          include/Align/AlignableVolume.h
          include/Align/EventVertex.h
          include/Align/ResidualsController.h
          include/Align/ResidualsControllerFast.h
          include/Align/GeometricalConstraint.h
          include/Align/DOFStatistics.h
          include/Align/utils.h)
