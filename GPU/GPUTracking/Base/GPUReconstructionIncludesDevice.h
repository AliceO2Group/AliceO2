// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionIncludesDevice.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONINCLUDESDEVICE_H
#define GPURECONSTRUCTIONINCLUDESDEVICE_H

#include "GPUDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
}
} // namespace GPUCA_NAMESPACE
using namespace GPUCA_NAMESPACE::gpu;

#include "GPUTPCTrackParam.cxx"
#include "GPUTPCTrack.cxx"
#include "GPUTPCGrid.cxx"
#include "GPUTPCRow.cxx"
#include "GPUTPCTracker.cxx"

#include "GPUGeneralKernels.cxx"
#include "GPUErrors.cxx"

#include "GPUTPCTrackletSelector.cxx"
#include "GPUTPCNeighboursFinder.cxx"
#include "GPUTPCNeighboursCleaner.cxx"
#include "GPUTPCStartHitsFinder.cxx"
#include "GPUTPCStartHitsSorter.cxx"
#include "GPUTPCTrackletConstructor.cxx"
#include "GPUTPCGlobalTracking.cxx"

#if !defined(GPUCA_OPENCL1) && !defined(GPUCA_ALIROOT_LIB)
// Files for TPC Merger
#include "GPUTPCGMMerger.cxx"
#include "GPUTPCGMMergerGPU.cxx"
#include "GPUTPCGMSliceTrack.cxx"
#include "GPUTPCGMTrackParam.cxx"
#include "GPUTPCGMPhysicalTrackModel.cxx"
#include "GPUTPCGMPropagator.cxx"
#include "GPUTPCSliceData.cxx"
#include "GPUTPCCreateSliceData.cxx"

#if defined(HAVE_O2HEADERS)
// Files for propagation with material
#include "MatLayerCylSet.cxx"
#include "MatLayerCyl.cxx"
#include "Ray.cxx"

// O2 track model
#include "TrackParametrization.cxx"
#include "TrackParametrizationWithError.cxx"
#include "Propagator.cxx"
#include "TrackLTIntegral.cxx"

// Files for GPU dEdx
#include "GPUdEdx.cxx"

// Files for TPC Transformation
#include "GPUTPCConvertKernel.cxx"

// Files for TPC Compression
#include "GPUTPCCompressionKernels.cxx"
#include "GPUTPCCompressionTrackModel.cxx"

// Files for TPC Cluster Finder
#include "ClusterAccumulator.cxx"
#include "GPUTPCCFStreamCompaction.cxx"
#include "GPUTPCCFChargeMapFiller.cxx"
#include "GPUTPCCFPeakFinder.cxx"
#include "GPUTPCCFNoiseSuppression.cxx"
#include "GPUTPCCFClusterizer.cxx"
#include "GPUTPCCFDeconvolution.cxx"
#include "GPUTPCCFMCLabelFlattener.cxx"
#include "GPUTPCCFCheckPadBaseline.cxx"
#include "GPUTPCCFDecodeZS.cxx"
#include "GPUTPCCFGather.cxx"

// Files for output into O2 format
#include "GPUTPCGMO2Output.cxx"

// Files for TRD Tracking
#include "GPUTRDTrackerKernels.cxx"
#include "GPUTRDTrack.cxx"
#include "GPUTRDTracker.cxx"
#include "GPUTRDTrackletWord.cxx"
#include "GeometryBase.cxx"

// Files for ITS Track Fit
#include "GPUITSFitterKernels.cxx"

// Files for Refit
#include "GPUTrackingRefit.cxx"
#include "GPUTrackingRefitKernel.cxx"

#if !defined(GPUCA_O2_LIB) && defined(__HIPCC__) && !defined(GPUCA_NO_ITS_TRAITS) && !defined(GPUCA_GPUCODE_GENRTC)
#include "VertexerTraitsHIP.hip.cxx"
#include "ContextHIP.hip.cxx"
#include "DeviceStoreVertexerHIP.hip.cxx"
#include "ClusterLinesHIP.hip.cxx"
#include "UtilsHIP.hip.cxx"
#elif !defined(GPUCA_O2_LIB) && defined(__CUDACC__) && !defined(GPUCA_NO_ITS_TRAITS) && !defined(GPUCA_GPUCODE_GENRTC)
#include "TrackerTraitsNV.cu"
#include "VertexerTraitsGPU.cu"
#include "Context.cu"
#include "Stream.cu"
#include "DeviceStoreNV.cu"
#include "DeviceStoreVertexerGPU.cu"
#include "ClusterLinesGPU.cu"
#include "Utils.cu"
#endif // !defined(GPUCA_O2_LIB) && defined(__CUDACC__) && !defined(GPUCA_NO_ITS_TRAITS)

#endif // HAVE_O2HEADERS
#endif // (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && !defined(GPUCA_ALIROOT_LIB)

#endif // GPURECONSTRUCTIONINCLUDESDEVICE_H
