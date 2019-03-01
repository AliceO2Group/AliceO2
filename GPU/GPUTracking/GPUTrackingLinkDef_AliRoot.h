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

#pragma link C++ class GPUTPCTrack+;
#pragma link C++ class GPUTPCTracklet+;
#pragma link C++ class GPUTPCTracker+;
#pragma link C++ class GPUTPCBaseTrackParam+;
#pragma link C++ class GPUTPCTrackParam+;
#pragma link C++ class GPUTPCRow+;
#pragma link C++ class GPUTPCGrid+;
#pragma link C++ class GPUTPCHitArea+;
#pragma link C++ class GPUTPCTrackerComponent+;
#pragma link C++ class GPUTPCNeighboursFinder+;
#pragma link C++ class GPUTPCNeighboursCleaner+;
#pragma link C++ class GPUTPCStartHitsFinder+;
#pragma link C++ class GPUTPCTrackletConstructor+;
#pragma link C++ class GPUTPCTrackletSelector+;
#pragma link C++ class GPUTPCGlobalMergerComponent+;
#pragma link C++ class GPUTPCClusterData+;
#pragma link C++ class GPUTPCSliceData+;
#pragma link C++ class GPUTPCSliceOutput+;
#pragma link C++ class GPUTPCGMTrackParam+;
#pragma link C++ class GPUTPCGMSliceTrack+;
#pragma link C++ class GPUTPCGMMerger+;
#pragma link C++ class GPUTPCGMPolynomialField+;
#pragma link C++ class GPUTPCGMPropagator+;
#pragma link C++ class GPUTPCGMPhysicalTrackModel+;
#pragma link C++ class GPUTPCGMPolynomialFieldManager+;
#pragma link C++ class AliHLTTPCClusterStatComponent+;

//#pragma link C++ class GPUTRDTrack+; //Templated, should add linkdef for specialization, but with an ifdef for ROOT >= 6 only
//#pragma link C++ class GPUTRDTracker+;
#pragma link C++ class GPUTRDTrackerComponent+;
//#pragma link C++ class GPUTRDTrackletWord+;
#pragma link C++ class GPUTRDTrackletReaderComponent+;

#endif
