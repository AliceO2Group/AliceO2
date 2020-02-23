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
GPUCA_KRNL((GPUTPCNeighboursFinder                       ), (single, REG, (GPUCA_THREAD_COUNT_FINDER, GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER_FINDER)), (), ())
GPUCA_KRNL((GPUTPCNeighboursCleaner                      ), (single, REG, (GPUCA_THREAD_COUNT, 1)), (), ())
GPUCA_KRNL((GPUTPCStartHitsFinder                        ), (single, REG, (GPUCA_THREAD_COUNT, 1)), (), ())
GPUCA_KRNL((GPUTPCStartHitsSorter                        ), (single, REG, (GPUCA_THREAD_COUNT, 1)), (), ())
GPUCA_KRNL((GPUTPCTrackletConstructor, singleSlice       ), (single, REG, (GPUCA_THREAD_COUNT_CONSTRUCTOR, GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER)), (), ())
GPUCA_KRNL((GPUTPCTrackletConstructor, allSlices         ), (single, REG, (GPUCA_THREAD_COUNT_CONSTRUCTOR, GPUCA_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER)), (), ())
GPUCA_KRNL((GPUTPCTrackletSelector                       ), (both, REG, (GPUCA_THREAD_COUNT_SELECTOR, GPUCA_BLOCK_COUNT_SELECTOR_MULTIPLIER)), (), ())
GPUCA_KRNL((GPUMemClean16                                ), (simple, REG, (GPUCA_THREAD_COUNT, 1)), (, GPUPtr1(void*, ptr), unsigned long size), (, GPUPtr2(void*, ptr), size))
#if !defined(GPUCA_OPENCL1) && (!defined(GPUCA_ALIROOT_LIB) || !defined(GPUCA_GPUCODE))
GPUCA_KRNL((GPUTPCGMMergerTrackFit                       ), (simple, REG, (GPUCA_THREAD_COUNT_FIT, 1)), (), ())
#ifdef HAVE_O2HEADERS
GPUCA_KRNL((GPUTRDTrackerGPU                             ), (simple, REG, (GPUCA_THREAD_COUNT_TRD, 1)), (), ())
GPUCA_KRNL((GPUITSFitterKernel                           ), (simple, REG, (GPUCA_THREAD_COUNT_ITS, 1)), (), ())
GPUCA_KRNL((GPUTPCConvertKernel                          ), (simple, REG, (GPUCA_THREAD_COUNT_CONVERTER, 1)), (), ())
GPUCA_KRNL((GPUTPCCompressionKernels,   step0attached    ), (simple, REG, (GPUCA_THREAD_COUNT_COMPRESSION1, 1)), (), ())
GPUCA_KRNL((GPUTPCCompressionKernels,   step1unattached  ), (simple, REG, (GPUCA_THREAD_COUNT_COMPRESSION2, 1)), (), ())
GPUCA_KRNL((GPUTPCCFChargeMapFiller,    fillChargeMap    ), (single, REG, (GPUCA_THREAD_COUNT_CLUSTERER, 1)), (), ())
GPUCA_KRNL((GPUTPCCFChargeMapFiller,    resetMaps        ), (single, REG, (GPUCA_THREAD_COUNT_CLUSTERER, 1)), (), ())
GPUCA_KRNL((GPUTPCCFPeakFinder                           ), (single, REG, (GPUCA_THREAD_COUNT_CLUSTERER, 1)), (), ())
GPUCA_KRNL((GPUTPCCFNoiseSuppression,   noiseSuppression ), (single, REG, (GPUCA_THREAD_COUNT_CLUSTERER, 1)), (), ())
GPUCA_KRNL((GPUTPCCFNoiseSuppression,   updatePeaks      ), (single, REG, (GPUCA_THREAD_COUNT_CLUSTERER, 1)), (), ())
GPUCA_KRNL((GPUTPCCFDeconvolution                        ), (single, REG, (GPUCA_THREAD_COUNT_CLUSTERER, 1)), (), ())
GPUCA_KRNL((GPUTPCCFClusterizer                          ), (single, REG, (GPUCA_THREAD_COUNT_CLUSTERER, 1)), (), ())
GPUCA_KRNL((GPUTPCCFStreamCompaction,   nativeScanUpStart), (single, REG, (GPUCA_THREAD_COUNT_SCAN, 1)), (, int iBuf, int stage), (, iBuf, stage))
GPUCA_KRNL((GPUTPCCFStreamCompaction,   nativeScanUp     ), (single, REG, (GPUCA_THREAD_COUNT_SCAN, 1)), (, int iBuf, int nElems), (, iBuf, nElems))
GPUCA_KRNL((GPUTPCCFStreamCompaction,   nativeScanTop    ), (single, REG, (GPUCA_THREAD_COUNT_SCAN, 1)), (, int iBuf, int nElems), (, iBuf, nElems))
GPUCA_KRNL((GPUTPCCFStreamCompaction,   nativeScanDown   ), (single, REG, (GPUCA_THREAD_COUNT_SCAN, 1)), (, int iBuf, unsigned int offset, int nElems), (, iBuf, offset, nElems))
GPUCA_KRNL((GPUTPCCFStreamCompaction,   compactDigit     ), (single, REG, (GPUCA_THREAD_COUNT_SCAN, 1)), (, int iBuf, int stage, GPUPtr1(deprecated::PackedDigit*, in), GPUPtr1(deprecated::PackedDigit*, out)), (, iBuf, stage, GPUPtr2(deprecated::PackedDigit*, in), GPUPtr2(deprecated::PackedDigit*, out)))
GPUCA_KRNL((GPUTPCCFDecodeZS                             ), (single, REG, (GPUCA_THREAD_COUNT_CFDECODE, 1)), (), ())
#endif
#endif
// clang-format on
