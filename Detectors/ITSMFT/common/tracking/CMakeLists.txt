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

o2_add_library(ITSMFTTrackingParams
               SOURCES src/TrackingConfigParam.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonUtils
                                     O2::DetectorsCommonDataFormats)

o2_target_root_dictionary(ITSMFTTrackingParams
                          HEADERS include/ITSMFTTracking/TrackingConfigParam.h
                          LINKDEF src/ITSMFTTrackingLinkDef.h)

o2_add_library(ITSMFTTracking
               TARGETVARNAME targetName
               SOURCES src/BoundedAllocator.cxx
                       src/CapacityEstimator.cxx
                       src/ITSTrackingConfigParam.cxx
                       src/SlabBumpAllocator.cxx
                       src/IOUtils.cxx
                       src/Propagator.cxx
                       src/Configuration.cxx
                       src/TimeFrame.cxx
                       src/TimeFrameScratch.cxx
                       src/TrackerTraits.cxx
                       src/CandidateFinding.cxx
                       src/TrackerTraversalPreparation.cxx
                       src/TripletFitting.cxx
                       src/PropagatorBarrelOperations.cxx
                       src/PropagatorForwardOperations.cxx
                       src/MaterialPhysics.cxx
                       src/FamilyMaterialOperations.cxx
                       src/IndexTableConfiguration.cxx
                       src/TraversalTopology.cxx
                       src/Tracker.cxx
               PUBLIC_LINK_LIBRARIES
                       O2::ITSMFTTrackingParams
                       O2::GPUCommon
                       O2::CommonConstants
                       O2::CommonDataFormat
                       O2::DetectorsCommonDataFormats
                       O2::DataFormatsITSMFT
                       O2::DataFormatsITS
                       O2::ITSMFTBase
                       O2::CommonUtils
                       O2::DetectorsBase
                       O2::FrameworkLogger
                       O2::MathUtils
                       Microsoft.GSL::GSL
                       O2::SimulationDataFormat
                       O2::ReconstructionDataFormats
                       O2::DataFormatsCalibration
                       O2::DataFormatsMFT
                       TBB::tbb
               PRIVATE_LINK_LIBRARIES
                       O2::Framework
                       O2::FrameworkLogger
                       O2::ITSBase
                       O2::MFTBase
                       O2::MFTTracking)

o2_target_root_dictionary(ITSMFTTracking
                          HEADERS include/ITSMFTTracking/ITSTrackingConfigParam.h
                          LINKDEF src/ITSTrackingLinkDef.h)

add_subdirectory(test)
