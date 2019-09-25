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
#include "GPUTPCHitArea.cxx"
#include "GPUTPCGrid.cxx"
#include "GPUTPCRow.cxx"
#include "GPUParam.cxx"
#include "GPUTPCTracker.cxx"

#include "GPUGeneralKernels.cxx"

#include "GPUTPCTrackletSelector.cxx"
#include "GPUTPCNeighboursFinder.cxx"
#include "GPUTPCNeighboursCleaner.cxx"
#include "GPUTPCStartHitsFinder.cxx"
#include "GPUTPCStartHitsSorter.cxx"
#include "GPUTPCTrackletConstructor.cxx"

#ifdef GPUCA_BUILD_MERGER
#include "GPUTPCGMMergerGPU.cxx"
#include "GPUTPCGMMerger.h"
#include "GPUTPCGMTrackParam.cxx"
#include "GPUTPCGMPhysicalTrackModel.cxx"
#include "GPUTPCGMPropagator.cxx"
#ifdef HAVE_O2HEADERS
#include "MatLayerCylSet.cxx"
#include "MatLayerCyl.cxx"
#include "Ray.cxx"
#endif
#endif

#ifdef GPUCA_BUILD_DEDX
#include "GPUdEdx.cxx"
#endif

#ifdef GPUCA_BUILD_TPCCONVERT
#include "GPUTPCConvertKernel.cxx"
#endif

#ifdef GPUCA_BUILD_TPCCOMPRESSION
#include "GPUTPCCompressionKernels.cxx"
#include "GPUTPCCompressionTrackModel.cxx"
#endif

#ifdef GPUCA_BUILD_TRD
#include "GPUTRDTrackerGPU.cxx"
#include "GPUTRDTrack.cxx"
#include "GPUTRDTracker.cxx"
#include "GPUTRDTrackletWord.cxx"
#include "TRDGeometryBase.cxx"
#endif

#ifdef GPUCA_BUILD_ITS
#include "GPUITSFitterKernels.cxx"
#if !defined(GPUCA_O2_LIB) && defined(__CUDACC__)
#include "TrackerTraitsNV.cu"
#include "VertexerTraitsGPU.cu"
#include "Context.cu"
#include "Stream.cu"
#include "DeviceStoreNV.cu"
#include "Utils.cu"
#endif
#endif

#endif
