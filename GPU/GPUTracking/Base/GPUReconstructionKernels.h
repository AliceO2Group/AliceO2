// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionKernels.h
/// \author David Rohr

// No header protection, this may be used multiple times
#include "GPUReconstructionKernelMacros.h"

// clang-format off
GPUCA_KRNL((GPUTPCNeighboursFinder                       ), (),       (), ())
GPUCA_KRNL((GPUTPCNeighboursCleaner                      ), (),       (), ())
GPUCA_KRNL((GPUTPCStartHitsFinder                        ), (),       (), ())
GPUCA_KRNL((GPUTPCStartHitsSorter                        ), (),       (), ())
GPUCA_KRNL((GPUTPCTrackletConstructor, singleSlice       ), (single), (), ())
GPUCA_KRNL((GPUTPCTrackletConstructor, allSlices         ), (multi),  (), ())
GPUCA_KRNL((GPUTPCTrackletSelector                       ), (both),   (), ())
GPUCA_KRNL((GPUMemClean16                                ), (),       (, void* ptr, unsigned long size), (, ptr, size))
GPUCA_KRNL((GPUTPCGMMergerTrackFit                       ), (),       (), ())
#ifdef HAVE_O2HEADERS
GPUCA_KRNL((GPUTRDTrackerGPU                             ), (),       (), ())
GPUCA_KRNL((GPUITSFitterKernel                           ), (),       (), ())
GPUCA_KRNL((GPUTPCConvertKernel                          ), (),       (), ())
GPUCA_KRNL((GPUTPCCompressionKernels,   step0attached    ), (),       (), ())
GPUCA_KRNL((GPUTPCCompressionKernels,   step1unattached  ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, fillChargeMap    ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, resetMaps        ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, findPeaks        ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, noiseSuppression ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, updatePeaks      ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, countPeaks       ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, computeClusters  ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, nativeScanUpStart), (),       (, int iBuf), (, iBuf))
GPUCA_KRNL((GPUTPCClusterFinderKernels, nativeScanUp     ), (),       (, int iBuf), (, iBuf))
GPUCA_KRNL((GPUTPCClusterFinderKernels, nativeScanTop    ), (),       (, int iBuf), (, iBuf))
GPUCA_KRNL((GPUTPCClusterFinderKernels, nativeScanDown   ), (),       (, int iBuf, unsigned int offset), (, iBuf, offset))
GPUCA_KRNL((GPUTPCClusterFinderKernels, compactDigit     ), (),       (, int iBuf, int stage, gpucf::PackedDigit* in, gpucf::PackedDigit* out), (, iBuf, stage, in, out))
#endif
// clang-format on
