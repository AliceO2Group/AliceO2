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
GPUCA_KRNL((GPUTPCNeighboursFinder                       ), (single), (), ())
GPUCA_KRNL((GPUTPCNeighboursCleaner                      ), (single), (), ())
GPUCA_KRNL((GPUTPCStartHitsFinder                        ), (single), (), ())
GPUCA_KRNL((GPUTPCStartHitsSorter                        ), (single), (), ())
GPUCA_KRNL((GPUTPCTrackletConstructor, singleSlice       ), (single), (), ())
GPUCA_KRNL((GPUTPCTrackletConstructor, allSlices         ), (single), (), ())
GPUCA_KRNL((GPUTPCTrackletSelector                       ), (both),   (), ())
GPUCA_KRNL((GPUMemClean16                                ), (),       (, GPUPtr1(void*, ptr), unsigned long size), (, GPUPtr2(void*, ptr), size))
#ifndef GPUCA_OPENCL1
GPUCA_KRNL((GPUTPCGMMergerTrackFit                       ), (),       (), ())
#ifdef HAVE_O2HEADERS
GPUCA_KRNL((GPUTRDTrackerGPU                             ), (),       (), ())
GPUCA_KRNL((GPUITSFitterKernel                           ), (),       (), ())
GPUCA_KRNL((GPUTPCConvertKernel                          ), (),       (), ())
GPUCA_KRNL((GPUTPCCompressionKernels,   step0attached    ), (),       (), ())
GPUCA_KRNL((GPUTPCCompressionKernels,   step1unattached  ), (),       (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, fillChargeMap    ), (single), (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, resetMaps        ), (single), (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, findPeaks        ), (single), (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, noiseSuppression ), (single), (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, updatePeaks      ), (single), (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, countPeaks       ), (single), (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, computeClusters  ), (single), (), ())
GPUCA_KRNL((GPUTPCClusterFinderKernels, nativeScanUpStart), (single), (, int iBuf), (, iBuf))
GPUCA_KRNL((GPUTPCClusterFinderKernels, nativeScanUp     ), (single), (, int iBuf), (, iBuf))
GPUCA_KRNL((GPUTPCClusterFinderKernels, nativeScanTop    ), (single), (, int iBuf), (, iBuf))
GPUCA_KRNL((GPUTPCClusterFinderKernels, nativeScanDown   ), (single), (, int iBuf, unsigned int offset), (, iBuf, offset))
GPUCA_KRNL((GPUTPCClusterFinderKernels, compactDigit     ), (single), (, int iBuf, int stage, GPUPtr1(deprecated::PackedDigit*, in), GPUPtr1(deprecated::PackedDigit*, out)), (, iBuf, stage, GPUPtr2(deprecated::PackedDigit*, in), GPUPtr2(deprecated::PackedDigit*, out)))
#endif
#endif
// clang-format on
