// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackingLinkDef_O2_DataTypes.h
/// \author David Rohr

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::gpu::trackInterface < o2::track::TrackParCov> + ;
#pragma link C++ class o2::gpu::GPUTRDTrack_t < o2::gpu::trackInterface < o2::track::TrackParCov>> + ;
#pragma link C++ class std::vector < o2::gpu::GPUTRDTrack_t < o2::gpu::trackInterface < o2::track::TrackParCov>>> + ;
#ifdef GPUCA_O2_LIB
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsO2 + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsRec + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsRecTPC + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsRecTRD + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsRecDynamic + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsProcessing + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsProcessingParam + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsProcessingRTC + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsProcessingRTCtechnical + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsProcessingNNclusterizer + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsDisplay + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsDisplayLight + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsDisplayHeavy + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsDisplayRenderer + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsDisplayVulkan + ;
#pragma link C++ class o2::gpu::internal::GPUConfigurableParamGPUSettingsQA + ;
#endif
#pragma link C++ class o2::gpu::GPUTPCGMMergedTrackHit + ;
#pragma link C++ class o2::tpc::CalibdEdxTrackTopologyPol + ;
#pragma link C++ class o2::tpc::CalibdEdxTrackTopologySpline + ;
#pragma link C++ struct o2::tpc::CalibdEdxTrackTopologyPolContainer + ;
#pragma link C++ struct o2::tpc::ORTRootSerializer + ;

#endif
