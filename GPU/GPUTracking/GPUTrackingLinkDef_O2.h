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

/// \file GPUTrackingLinkDef_O2.h
/// \author David Rohr

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::gpu::GPUTPCGMMergedTrack + ;
#pragma link C++ class o2::gpu::GPUTPCGMSliceTrack + ;
#pragma link C++ class o2::gpu::GPUTPCGMBorderTrack + ;
#pragma link C++ class o2::gpu::GPUTPCGMTrackParam + ;
#pragma link C++ class o2::gpu::GPUTPCTrack + ;
#pragma link C++ struct o2::gpu::GPUTPCBaseTrackParam + ;
#pragma link C++ struct o2::gpu::GPUTPCGMSliceTrack::sliceTrackParam + ;
#pragma link C++ class o2::gpu::gputpcgmmergertypes::GPUTPCOuterParam + ;
#pragma link C++ class o2::gpu::gputpcgmmergertypes::InterpolationErrorHit + ;

#endif
