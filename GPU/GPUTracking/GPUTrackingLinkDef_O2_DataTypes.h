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
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsO2 + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsRec + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsRecTPC + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsRecTRD + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsProcessing + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsProcessingRTC + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsDisplay + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsDisplayLight + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsDisplayHeavy + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsDisplayRenderer + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsDisplayVulkan + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsQA + ;
#pragma link C++ class o2::tpc::CalibdEdxTrackTopologyPol + ;
#pragma link C++ class o2::tpc::CalibdEdxTrackTopologySpline + ;
#pragma link C++ struct o2::tpc::CalibdEdxTrackTopologyPolContainer + ;

#endif
