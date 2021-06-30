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

/// \file GPUTRDDef.h
/// \author David Rohr

#ifndef GPUTRDDEF_H
#define GPUTRDDEF_H

#include "GPUCommonDef.h"

#ifdef GPUCA_ALIROOT_LIB
#define TRD_TRACK_TYPE_ALIROOT
#else
#define TRD_TRACK_TYPE_O2
#endif

#ifdef GPUCA_ALIROOT_LIB
class AliExternalTrackParam;
class AliTrackerBase;
#else
namespace o2
{
namespace gpu
{
class GPUTRDO2BaseTrack;
} // namespace gpu
namespace base
{
template <typename>
class PropagatorImpl;
} // namespace base
} // namespace o2
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

#ifdef GPUCA_ALIROOT_LIB
typedef double My_Float;
#else
typedef float My_Float;
#endif

#if defined(TRD_TRACK_TYPE_ALIROOT)
typedef AliExternalTrackParam TRDBaseTrack;
class GPUTPCGMTrackParam;
typedef GPUTPCGMTrackParam TRDBaseTrackGPU;
#elif defined(TRD_TRACK_TYPE_O2)
typedef o2::gpu::GPUTRDO2BaseTrack TRDBaseTrack;
class GPUTPCGMTrackParam;
typedef GPUTPCGMTrackParam TRDBaseTrackGPU;
#endif

#ifdef GPUCA_ALIROOT_LIB
typedef AliTrackerBase TRDBasePropagator;
class GPUTPCGMPropagator;
typedef GPUTPCGMPropagator TRDBasePropagatorGPU;
#else
typedef o2::base::PropagatorImpl<float> TRDBasePropagator;
class GPUTPCGMPropagator;
typedef GPUTPCGMPropagator TRDBasePropagatorGPU;
#endif

template <class T>
class trackInterface;
template <class T>
class propagatorInterface;
template <class T>
class GPUTRDTrack_t;
// clang-format off
typedef GPUTRDTrack_t<trackInterface<TRDBaseTrack> > GPUTRDTrack; // Need pre-c++11 compliant formatting
typedef GPUTRDTrack_t<trackInterface<TRDBaseTrackGPU> > GPUTRDTrackGPU;
// clang-foramt on
typedef propagatorInterface<TRDBasePropagator> GPUTRDPropagator;
typedef propagatorInterface<TRDBasePropagatorGPU> GPUTRDPropagatorGPU;

template <class T, class P>
class GPUTRDTracker_t;
typedef GPUTRDTracker_t<GPUTRDTrack, GPUTRDPropagator> GPUTRDTracker;
typedef GPUTRDTracker_t<GPUTRDTrackGPU, GPUTRDPropagatorGPU> GPUTRDTrackerGPU;

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTRDDEF_H
