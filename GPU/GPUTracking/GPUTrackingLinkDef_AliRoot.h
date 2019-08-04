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

#pragma link C++ class AliGPU::gpu::GPUTPCTrack + ;
#pragma link C++ class AliGPU::gpu::GPUTPCTracklet + ;
#pragma link C++ class AliGPU::gpu::GPUTPCBaseTrackParam + ;
#pragma link C++ class AliGPU::gpu::GPUTPCTrackParam + ;
#pragma link C++ class AliGPU::gpu::GPUTPCRow + ;
#pragma link C++ class AliGPU::gpu::GPUTPCGrid + ;
#pragma link C++ class AliGPU::gpu::GPUTPCHitArea + ;
#pragma link C++ class GPUTPCTrackerComponent + ;
#pragma link C++ class AliGPU::gpu::GPUTPCNeighboursFinder + ;
#pragma link C++ class AliGPU::gpu::GPUTPCNeighboursCleaner + ;
#pragma link C++ class AliGPU::gpu::GPUTPCStartHitsFinder + ;
#pragma link C++ class AliGPU::gpu::GPUTPCTrackletConstructor + ;
#pragma link C++ class AliGPU::gpu::GPUTPCTrackletSelector + ;
#pragma link C++ class GPUTPCGlobalMergerComponent + ;
#pragma link C++ class AliGPU::gpu::GPUTPCClusterData + ;
#pragma link C++ class AliGPU::gpu::GPUTPCSliceData + ;
#pragma link C++ class AliGPU::gpu::GPUTPCSliceOutput + ;
#pragma link C++ class AliGPU::gpu::GPUTPCGMTrackParam + ;
#pragma link C++ class AliGPU::gpu::GPUTPCGMSliceTrack + ;
#pragma link C++ class AliGPU::gpu::GPUTPCGMPolynomialField + ;
#pragma link C++ class AliGPU::gpu::GPUTPCGMPropagator + ;
#pragma link C++ class AliGPU::gpu::GPUTPCGMPhysicalTrackModel + ;
#pragma link C++ class GPUTPCGMPolynomialFieldManager + ;
#pragma link C++ class AliHLTTPCClusterStatComponent + ;

//#pragma link C++ class AliGPU::gpu::GPUTRDTrack+; //Templated, should add linkdef for specialization, but with an ifdef for ROOT >= 6 only
//#pragma link C++ class AliGPU::gpu::GPUTRDTracker+;
#pragma link C++ class GPUTRDTrackerComponent + ;
//#pragma link C++ class AliGPU::gpu::GPUTRDTrackletWord+;
#pragma link C++ class GPUTRDTrackletReaderComponent + ;

#endif
