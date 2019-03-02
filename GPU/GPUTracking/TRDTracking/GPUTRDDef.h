#ifndef GPUTRDDEF_H
#define GPUTRDDEF_H

class AliExternalTrackParam;
class AliTrackerBase;

namespace o2
{
namespace gpu
{

#ifdef GPUCA_ALIROOT_LIB
typedef double My_Float;
#else
typedef float My_Float;
#endif

#ifdef GPUCA_ALIROOT_LIB
#define TRD_TRACK_TYPE_ALIROOT
#else
#define TRD_TRACK_TYPE_O2
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

#if !defined(GPUCA_ALIROOT_LIB) && !defined(__CLING__) && !defined(__ROOTCLING__)
#define Error(...)
#define Warning(...)
#define Info(...)
#endif
} // namespace gpu
} // namespace o2

#endif
