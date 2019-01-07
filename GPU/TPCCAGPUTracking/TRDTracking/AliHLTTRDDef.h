#ifndef ALIHLTTRDDEF_H
#define ALIHLTTRDDEF_H

#ifdef GPUCA_BUILD_ALIROOT_LIB
typedef double My_Float;
#else
typedef float My_Float;
#endif

#ifdef GPUCA_BUILD_ALIROOT_LIB
#define TRD_TRACK_TYPE_ALIROOT
#else
#define TRD_TRACK_TYPE_HLT
#endif

#if defined (TRD_TRACK_TYPE_ALIROOT)
class AliExternalTrackParam;
typedef AliExternalTrackParam HLTTRDBaseTrack;
//class AliHLTTPCGMTrackParam;
//typedef AliHLTTPCGMTrackParam HLTTRDBaseTrack;
#elif defined (TRD_TRACK_TYPE_HLT)
class AliHLTTPCGMTrackParam;
typedef AliHLTTPCGMTrackParam HLTTRDBaseTrack;
#endif

#ifdef GPUCA_BUILD_ALIROOT_LIB
class AliTrackerBase;
typedef AliTrackerBase HLTTRDBasePropagator;
//class AliHLTTPCGMPropagator;
//typedef AliHLTTPCGMPropagator HLTTRDBasePropagator;
#else
class AliHLTTPCGMPropagator;
typedef AliHLTTPCGMPropagator HLTTRDBasePropagator;
#endif

template <class T> class trackInterface;
template <class T> class propagatorInterface;
template <class T> class AliHLTTRDTrack;
typedef AliHLTTRDTrack<trackInterface<HLTTRDBaseTrack>> HLTTRDTrack;
typedef propagatorInterface<HLTTRDBasePropagator> HLTTRDPropagator;

#if defined(GPUCA_BUILD_ALIROOT_LIB) || defined(__CLING__) || defined(__ROOTCLING__)
#include "TMath.h"
#else
#define Error(...)
#define Warning(...)
#define Info(...)
#endif

#endif
