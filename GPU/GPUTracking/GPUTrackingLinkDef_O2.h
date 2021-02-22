// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackingLinkDef_O2.h
/// \author David Rohr

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::gpu::TPCdEdxCalibrationSplines + ;
#pragma link C++ class o2::gpu::trackInterface < o2::dataformats::TrackTPCITS> + ;
#pragma link C++ class o2::gpu::GPUTRDTrack_t < o2::gpu::trackInterface < o2::dataformats::TrackTPCITS>> + ;
#pragma link C++ class std::vector < o2::gpu::GPUTRDTrack_t < o2::gpu::trackInterface < o2::dataformats::TrackTPCITS>>> + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsO2 + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsRec + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsProcessing + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsDisplay + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsDisplayLight + ;
#pragma link C++ class o2::gpu::GPUConfigurableParamGPUSettingsQA + ;

#endif
