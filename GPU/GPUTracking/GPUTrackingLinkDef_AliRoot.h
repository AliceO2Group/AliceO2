// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackingLinkDef_AliRoot.h
/// \author David Rohr

#if defined(__CINT__) || defined(__CLING__)

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::gpu::GPUTPCTrack+;
#pragma link C++ class o2::gpu::GPUTPCTracklet+;
#pragma link C++ class o2::gpu::GPUTPCBaseTrackParam+;
#pragma link C++ class o2::gpu::GPUTPCTrackParam+;
#pragma link C++ class o2::gpu::GPUTPCRow+;
#pragma link C++ class o2::gpu::GPUTPCGrid+;
#pragma link C++ class o2::gpu::GPUTPCHitArea+;
#pragma link C++ class GPUTPCTrackerComponent+;
#pragma link C++ class o2::gpu::GPUTPCNeighboursFinder+;
#pragma link C++ class o2::gpu::GPUTPCNeighboursCleaner+;
#pragma link C++ class o2::gpu::GPUTPCStartHitsFinder+;
#pragma link C++ class o2::gpu::GPUTPCTrackletConstructor+;
#pragma link C++ class o2::gpu::GPUTPCTrackletSelector+;
#pragma link C++ class GPUTPCGlobalMergerComponent+;
#pragma link C++ class o2::gpu::GPUTPCClusterData+;
#pragma link C++ class o2::gpu::GPUTPCSliceData+;
#pragma link C++ class o2::gpu::GPUTPCSliceOutput+;
#pragma link C++ class o2::gpu::GPUTPCGMTrackParam+;
#pragma link C++ class o2::gpu::GPUTPCGMSliceTrack+;
#pragma link C++ class o2::gpu::GPUTPCGMPolynomialField+;
#pragma link C++ class o2::gpu::GPUTPCGMPropagator+;
#pragma link C++ class o2::gpu::GPUTPCGMPhysicalTrackModel+;
#pragma link C++ class GPUTPCGMPolynomialFieldManager+;
#pragma link C++ class AliHLTTPCClusterStatComponent+;

//#pragma link C++ class o2::gpu::GPUTRDTrack+; //Templated, should add linkdef for specialization, but with an ifdef for ROOT >= 6 only
//#pragma link C++ class o2::gpu::GPUTRDTracker+;
#pragma link C++ class GPUTRDTrackerComponent+;
//#pragma link C++ class o2::gpu::GPUTRDTrackletWord+;
#pragma link C++ class GPUTRDTrackletReaderComponent+;

#endif
