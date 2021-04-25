// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCPadGainCalib.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_TPC_PAD_GAIN_CALIB_H
#define O2_GPU_TPC_PAD_GAIN_CALIB_H

#include "clusterFinderDefs.h"
#include "GPUCommonMath.h"

namespace o2::tpc
{
template <class T>
class CalDet;
} // namespace o2::tpc

namespace GPUCA_NAMESPACE::gpu
{

template <typename T>
struct TPCPadGainCorrectionStepNum {
};

template <>
struct TPCPadGainCorrectionStepNum<unsigned char> {
  static constexpr int value = 254;
};

template <>
struct TPCPadGainCorrectionStepNum<unsigned short> {
  static constexpr int value = 65534;
};

struct TPCPadGainCalib {
 public:
#ifndef GPUCA_GPUCODE
  TPCPadGainCalib();
  TPCPadGainCalib(const o2::tpc::CalDet<float>&);
#endif

  // Deal with pad gain correction from here on
  GPUdi() void setGainCorrection(int sector, tpccf::Row row, tpccf::Pad pad, float c)
  {
    mGainCorrection[sector].set(globalPad(row, pad), c);
  }

  GPUdi() float getGainCorrection(int sector, tpccf::Row row, tpccf::Pad pad) const
  {
    return mGainCorrection[sector].get(globalPad(row, pad));
  }

  GPUdi() unsigned short globalPad(tpccf::Row row, tpccf::Pad pad) const
  {
    return mPadOffsetPerRow[row] + pad;
  }

 private:
  template <typename T = unsigned short>
  class SectorPadGainCorrection
  {

   public:
    constexpr static float MinCorrectionFactor = 0.f;
    constexpr static float MaxCorrectionFactor = 2.f;
    constexpr static int NumOfSteps = TPCPadGainCorrectionStepNum<T>::value;

    GPUdi() SectorPadGainCorrection()
    {
      reset();
    }

    GPUdi() void set(unsigned short globalPad, float c)
    {
      at(globalPad) = pack(c);
    }

    GPUdi() float get(unsigned short globalPad) const
    {
      return unpack(at(globalPad));
    }

    GPUd() void reset()
    {
      for (unsigned short p = 0; p < TPC_PADS_IN_SECTOR; p++) {
        set(p, 1.0f);
      }
    }

   private:
    GPUd() static T pack(float f)
    {
      f = CAMath::Clamp(f, MinCorrectionFactor, MaxCorrectionFactor);
      f -= MinCorrectionFactor;
      f *= float(NumOfSteps);
      f /= (MaxCorrectionFactor - MinCorrectionFactor);
      return CAMath::Nint(f);
    }

    GPUd() static float unpack(T c)
    {
      return MinCorrectionFactor + (MaxCorrectionFactor - MinCorrectionFactor) * float(c) / float(NumOfSteps);
    }

    T mGainCorrection[TPC_PADS_IN_SECTOR];

    GPUdi() T& at(unsigned short globalPad)
    {
      return mGainCorrection[globalPad];
    }

    GPUdi() const T& at(unsigned short globalPad) const
    {
      return mGainCorrection[globalPad];
    }
  };

  unsigned short mPadOffsetPerRow[GPUCA_ROW_COUNT];
  SectorPadGainCorrection<unsigned short> mGainCorrection[GPUCA_NSLICES];
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
