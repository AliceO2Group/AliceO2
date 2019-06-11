// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
// class GPUTPCGMTrackParam;
// typedef GPUTPCGMTrackParam TRDBaseTrack;
#elif defined(TRD_TRACK_TYPE_O2)
class GPUTPCGMTrackParam;
typedef GPUTPCGMTrackParam TRDBaseTrack;
#endif

#ifdef GPUCA_ALIROOT_LIB
typedef AliTrackerBase TRDBasePropagator;
// class GPUTPCGMPropagator;
// typedef GPUTPCGMPropagator TRDBasePropagator;
#else
class GPUTPCGMPropagator;
typedef GPUTPCGMPropagator TRDBasePropagator;
#endif

template <class T>
class trackInterface;
template <class T>
class propagatorInterface;
template <class T>
class GPUTRDTrack_t;
typedef GPUTRDTrack_t<trackInterface<TRDBaseTrack>> GPUTRDTrack;
typedef propagatorInterface<TRDBasePropagator> GPUTRDPropagator;

#if defined(GPUCA_ALIGPUCODE) && !defined(GPUCA_ALIROOT_LIB) && !defined(__CLING__) && !defined(__ROOTCLING__) && !defined(G__ROOT)
#define Error(...)
#define Warning(...)
#define Info(...)
#endif
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTRDDEF_H
