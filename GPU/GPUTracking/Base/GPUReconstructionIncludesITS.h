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

/// \file GPUReconstructionIncludesITS.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONINCLDUESITS_H
#define GPURECONSTRUCTIONINCLDUESITS_H

#if defined(GPUCA_HAVE_O2HEADERS) && !defined(GPUCA_STANDALONE)
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/TimeFrame.h"
#if defined(__CUDACC__) || defined(__HIPCC__)
#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/VertexerTraitsGPU.h"
#include "ITStrackingGPU/TimeFrameGPU.h"
#endif
#else
namespace o2::its
{
class VertexerTraits
{
};
class TrackerTraits
{
};
class TimeFrame
{
};
class VertexerTraitsGPU : public VertexerTraits
{
};
template <int NLayers = 7>
class TrackerTraitsGPU : public TrackerTraits
{
};
namespace gpu
{
template <int NLayers = 7>
class TimeFrameGPU : public TimeFrame
{
};
} // namespace gpu
} // namespace o2::its
#endif

#endif
