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
struct TPCPadGainCorrectionStepNum<uint8_t> {
  static constexpr int32_t value = 254;
};

template <>
struct TPCPadGainCorrectionStepNum<uint16_t> {
  static constexpr int32_t value = 65534;
};

struct TPCPadGainCalib {
 public:
#ifndef GPUCA_GPUCODE
  TPCPadGainCalib();
  TPCPadGainCalib(const o2::tpc::CalDet<float>&);

  /// constructor
  /// \param minValue minimum value which will be stored
  /// \param maxValue maximum value which will be stored
  /// \param inv setting the inverse value
  TPCPadGainCalib(const o2::tpc::CalDet<float>&, const float minValue, const float maxValue, const bool inv);

  /// setting the stored values from CalDet
  /// \param inv setting the inverse value
  void setFromMap(const o2::tpc::CalDet<float>&, const bool inv = true);
#endif

  // Deal with pad gain correction from here on
  GPUdi() void setGainCorrection(int32_t sector, tpccf::Row row, tpccf::Pad pad, float c)
  {
    mGainCorrection[sector].set(globalPad(row, pad), c);
  }

  GPUdi() void setGainCorrection(int32_t sector, uint16_t globalPad, float c)
  {
    mGainCorrection[sector].set(globalPad, c);
  }

  GPUdi() float getGainCorrection(int32_t sector, tpccf::Row row, tpccf::Pad pad) const
  {
    return mGainCorrection[sector].get(globalPad(row, pad));
  }

  GPUdi() uint16_t globalPad(tpccf::Row row, tpccf::Pad pad) const
  {
    return mPadOffsetPerRow[row] + pad;
  }

  GPUdi() void setMinCorrectionFactor(const float minCorrectionFactor)
  {
    for (int32_t sector = 0; sector < GPUCA_NSLICES; sector++) {
      mGainCorrection[sector].mMinCorrectionFactor = minCorrectionFactor;
    }
  }

  GPUdi() void setMaxCorrectionFactor(const float maxCorrectionFactor)
  {
    for (int32_t sector = 0; sector < GPUCA_NSLICES; sector++) {
      mGainCorrection[sector].mMaxCorrectionFactor = maxCorrectionFactor;
    }
  }

 private:
  template <typename T = uint16_t>
  class SectorPadGainCorrection
  {

   public:
    float mMinCorrectionFactor = 0.f;
    float mMaxCorrectionFactor = 2.f;
    constexpr static int32_t NumOfSteps = TPCPadGainCorrectionStepNum<T>::value;

    GPUdi() SectorPadGainCorrection()
    {
      reset();
    }

    GPUdi() void set(uint16_t globalPad, float c)
    {
      at(globalPad) = pack(c);
    }

    GPUdi() float get(uint16_t globalPad) const
    {
      return unpack(at(globalPad));
    }

    GPUd() void reset()
    {
      for (uint16_t p = 0; p < TPC_PADS_IN_SECTOR; p++) {
        set(p, 1.0f);
      }
    }

   private:
    GPUd() T pack(float f) const
    {
      f = CAMath::Clamp(f, mMinCorrectionFactor, mMaxCorrectionFactor);
      f -= mMinCorrectionFactor;
      f *= float(NumOfSteps);
      f /= (mMaxCorrectionFactor - mMinCorrectionFactor);
      return CAMath::Round(f);
    }

    GPUd() float unpack(T c) const
    {
      return mMinCorrectionFactor + (mMaxCorrectionFactor - mMinCorrectionFactor) * float(c) / float(NumOfSteps);
    }

    T mGainCorrection[TPC_PADS_IN_SECTOR];

    GPUdi() T& at(uint16_t globalPad)
    {
      return mGainCorrection[globalPad];
    }

    GPUdi() const T& at(uint16_t globalPad) const
    {
      return mGainCorrection[globalPad];
    }
  };

  uint16_t mPadOffsetPerRow[GPUCA_ROW_COUNT];
  SectorPadGainCorrection<uint16_t> mGainCorrection[GPUCA_NSLICES];
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
