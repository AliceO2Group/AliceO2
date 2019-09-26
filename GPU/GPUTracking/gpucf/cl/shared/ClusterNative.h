// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#if !defined(CLUSTER_NATIVE_H)
#define CLUSTER_NATIVE_H

#include "types.h"

#define CN_SCALE_TIME_PACKED 64
#define CN_SCALE_PAD_PACKED 64
#define CN_SCALE_SIGMA_TIME_PACKED 32
#define CN_SCALE_SIGMA_PAD_PACKED 32

#define CN_TIME_MASK 0xFFFFFF
#define CN_FLAG_MASK 0xFF000000

#if IS_CL_DEVICE
#define CAST(type, val) ((type)(val))
#else
#define CAST(type, val) static_cast<type>(val)
#endif

typedef struct ClusterNative_s {
  SHARED_UINT timeFlagsPacked;
  SHARED_USHORT padPacked;
  SHARED_UCHAR sigmaTimePacked;
  SHARED_UCHAR sigmaPadPacked;
  SHARED_USHORT qmax;
  SHARED_USHORT qtot;
} ClusterNative;

enum CnFlagPos {
  CN_FLAG_POS_IS_EDGE_CLUSTER = 0,
  CN_FLAG_POS_SPLIT_IN_TIME = 1,
  CN_FLAG_POS_SPLIT_IN_PAD = 2,
};

enum CnFlag {
  CN_FLAG_IS_EDGE_CLUSTER = (1 << CN_FLAG_POS_IS_EDGE_CLUSTER),
  CN_FLAG_SPLIT_IN_TIME = (1 << CN_FLAG_POS_SPLIT_IN_TIME),
  CN_FLAG_SPLIT_IN_PAD = (1 << CN_FLAG_POS_SPLIT_IN_PAD),
};

inline SHARED_USHORT cnPackPad(SHARED_FLOAT pad)
{
  return CAST(SHARED_USHORT, pad * CN_SCALE_PAD_PACKED + 0.5f);
}

inline SHARED_UINT cnPackTime(SHARED_FLOAT time)
{
  return CAST(SHARED_UINT, time * CN_SCALE_TIME_PACKED + 0.5f);
}

inline SHARED_FLOAT cnUnpackPad(SHARED_USHORT pad)
{
  return CAST(SHARED_FLOAT, pad) * (1.f / CN_SCALE_PAD_PACKED);
}

inline SHARED_FLOAT cnUnpackTime(SHARED_UINT time)
{
  return CAST(SHARED_FLOAT, time) * (1.f / CN_SCALE_TIME_PACKED);
}

inline SHARED_UCHAR cnPackSigma(SHARED_FLOAT sigma, SHARED_FLOAT scale)
{
  SHARED_UINT tmp = sigma * scale + 0.5f;
  return (tmp > 0xFF) ? 0xFF : tmp;
}

inline SHARED_FLOAT cnUnpackSigma(
  SHARED_UCHAR sigmaPacked,
  SHARED_FLOAT scale)
{
  return CAST(SHARED_FLOAT, sigmaPacked) * (1.f / scale);
}

inline SHARED_UCHAR cnGetFlags(const ClusterNative* c)
{
  return c->timeFlagsPacked >> 24;
}

inline SHARED_UINT cnGetTimePacked(const ClusterNative* c)
{
  return c->timeFlagsPacked & CN_TIME_MASK;
}

inline void cnSetTimePackedFlags(
  ClusterNative* c,
  SHARED_UINT timePacked,
  SHARED_UCHAR flags)
{
  c->timeFlagsPacked = (timePacked & CN_TIME_MASK) | CAST(SHARED_UINT, flags) << 24;
}

inline void cnSetTimePacked(ClusterNative* c, SHARED_UINT timePacked)
{
  c->timeFlagsPacked = (timePacked & CN_TIME_MASK) | (c->timeFlagsPacked & CN_FLAG_MASK);
}

inline void cnSetFlags(ClusterNative* c, SHARED_UCHAR flags)
{
  c->timeFlagsPacked = (c->timeFlagsPacked & CN_TIME_MASK) | (CAST(SHARED_UINT, flags) << 24);
}

inline SHARED_FLOAT cnGetTime(const ClusterNative* c)
{
  return cnUnpackTime(c->timeFlagsPacked & CN_TIME_MASK);
}

inline void cnSetTime(ClusterNative* c, float time)
{
  c->timeFlagsPacked = (cnPackTime(time) & CN_TIME_MASK) | (c->timeFlagsPacked & CN_FLAG_MASK);
}

inline void cnSetTimeFlags(
  ClusterNative* c,
  SHARED_FLOAT time,
  SHARED_UCHAR flags)
{
  c->timeFlagsPacked = (cnPackTime(time) & CN_TIME_MASK) | (CAST(SHARED_UINT, flags) << 24);
}

inline SHARED_FLOAT cnGetPad(const ClusterNative* c)
{
  return cnUnpackPad(c->padPacked);
}

inline void cnSetPad(ClusterNative* c, SHARED_FLOAT pad)
{
  c->padPacked = cnPackPad(pad);
}

inline SHARED_FLOAT cnGetSigmaTime(const ClusterNative* c)
{
  return cnUnpackSigma(c->sigmaTimePacked, CN_SCALE_SIGMA_TIME_PACKED);
}

inline void cnSetSigmaTime(ClusterNative* c, SHARED_FLOAT sigmaTime)
{
  c->sigmaTimePacked = cnPackSigma(sigmaTime, CN_SCALE_SIGMA_TIME_PACKED);
}

inline SHARED_FLOAT cnGetSigmaPad(const ClusterNative* c)
{
  return cnUnpackSigma(c->sigmaPadPacked, CN_SCALE_SIGMA_PAD_PACKED);
}

inline void cnSetSigmaPad(ClusterNative* c, SHARED_FLOAT sigmaPad)
{
  c->sigmaPadPacked = cnPackSigma(sigmaPad, CN_SCALE_SIGMA_PAD_PACKED);
}

#endif // !defined(CLUSTER_NATIVE_H)
// vim: set ts=4 sw=4 sts=4 expandtab:
