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

/// \file CalibdEdxCorrection.h
/// \author Thiago Badaró <thiago.saramela@usp.br>

#ifndef ALICEO2_TPC_CALIBDEDXCORRECTION_H_
#define ALICEO2_TPC_CALIBDEDXCORRECTION_H_

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonRtypes.h"

#ifndef GPUCA_GPUCODE_DEVICE
#include <string_view>
#include <algorithm>
#endif

// o2 includes
#include "DataFormatsTPC/Defs.h"

namespace o2::tpc
{

class CalibdEdxCorrection
{
 public:
  static constexpr int FitSize = 288; ///< Number of fitted corrections
  static constexpr int ParamSize = 8; ///< Number of params per fit

#if !defined(GPUCA_ALIGPUCODE)
  CalibdEdxCorrection()
  {
    clear();
  }
  CalibdEdxCorrection(std::string_view fileName) { loadFromFile(fileName); }
#else
  CalibdEdxCorrection() CON_DEFAULT;
#endif
  ~CalibdEdxCorrection() CON_DEFAULT;

  GPUd() float getCorrection(const StackID& stack, ChargeType charge, float tgl = 0, float snp = 0) const
  {
    // by default return 1 if no correction was loaded
    if (mDims < 0) {
      return 1;
    }

    tgl = o2::gpu::CAMath::Abs(tgl);
    auto p = mParams[stackIndex(stack, charge)];
    float result = p[0];
    // Tgl part
    if (mDims > 0) {
      result += tgl * (p[1] + tgl * (p[2] + tgl * (p[3] + tgl * p[4])));
    }
    // Snp and cross terms
    if (mDims > 1) {
      result += snp * (p[5] + snp * p[6] + tgl * p[7]);
    }
    return result;
  }

#if !defined(GPUCA_GPUCODE)
  const float* getParams(const StackID& stack, ChargeType charge) const
  {
    return mParams[stackIndex(stack, charge)];
  }
  float getChi2(const StackID& stack, ChargeType charge) const { return mChi2[stackIndex(stack, charge)]; }
  int getEntries(const StackID& stack, ChargeType charge) const { return mEntries[stackIndex(stack, charge)]; }
  int getDims() const { return mDims; }

  void setParams(const StackID& stack, ChargeType charge, const float* params) { std::copy(params, params + ParamSize, mParams[stackIndex(stack, charge)]); }
  void setChi2(const StackID& stack, ChargeType charge, float chi2) { mChi2[stackIndex(stack, charge)] = chi2; }
  void setEntries(const StackID& stack, ChargeType charge, int entries) { mEntries[stackIndex(stack, charge)] = entries; }
  void setDims(int dims) { mDims = dims; }

  void clear();

  void writeToFile(std::string_view fileName) const;
  void loadFromFile(std::string_view fileName);

  /// \param outFileName name of the output file
  void dumpToTree(const char* outFileName = "calib_dedx.root") const;

#endif

 private:
  GPUd() static int stackIndex(const StackID& stack, ChargeType charge)
  {
    return stack.getIndex() + charge * SECTORSPERSIDE * SIDES * GEMSTACKSPERSECTOR;
  }

  float mParams[FitSize][ParamSize];
  float mChi2[FitSize];
  int mEntries[FitSize];
  int mDims{-1}; ///< Fit dimension

  ClassDefNV(CalibdEdxCorrection, 2);
};

} // namespace o2::tpc

#endif
