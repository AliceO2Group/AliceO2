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

#ifndef O2_TRD_COMMONPARAM_H
#define O2_TRD_COMMONPARAM_H

#include <array>
#include "Rtypes.h" // for ClassDef
#include "DataFormatsTRD/Constants.h"
#include "TRDBase/Garfield.h"

namespace o2
{
namespace trd
{

class CommonParam
{
 public:
  enum { kXenon = 0,
         kArgon = 1 };

  CommonParam(const CommonParam&) = delete;
  CommonParam& operator=(const CommonParam&) = delete;
  ~CommonParam() = default;

  static CommonParam* instance();

  void setExB(bool flag = true) { mExBOn = flag; }
  void setSamplingFrequency(float freq) { mSamplingFrequency = freq; }
  void setXenon();
  void setArgon();

  bool isExBOn() const { return mExBOn; }
  bool isXenon() const { return (mGasMixture == kXenon); }
  bool isArgon() const { return (mGasMixture == kArgon); }
  int getGasMixture() const { return mGasMixture; }
  float getSamplingFrequency() const { return mSamplingFrequency; }
  float getCachedField() const { return mField; }

  // Cached magnetic field, to be called by the user before using DiffusionAndTimeStructEstimator::GetDiffCoeff
  void cacheMagField();

 protected:

  static CommonParam* mgInstance;    ///<  Instance of this class (singleton implementation)
  bool mExBOn{true};                 ///< Switch for the ExB effects
  double mField{-0.5};               ///< Cached magnetic field
  float mDiffusionT{0.};             ///< Transverse drift coefficient
  float mDiffusionL{0.};             ///< Longitudinal drift coefficient
  float mDiffLastVdrift{-1.};        ///< The structures are valid for fLastVdrift (caching)

  std::array<float, garfield::TIMEBINSGARFIELD * garfield::ZBINSGARFIELD> mTimeStruct1{}; ///< Time Structure of Drift Cells
  std::array<float, garfield::TIMEBINSGARFIELD * garfield::ZBINSGARFIELD> mTimeStruct2{}; ///< Time Structure of Drift Cells
  float mVDlo{0.};                                  ///< Lower drift velocity, for interpolation
  float mVDhi{0.};                                  ///< Higher drift velocity, for interpolation
  float mTimeLastVdrift{-1.};                       ///< The structures are valid for fLastVdrift (caching)

  float mSamplingFrequency{10.}; ///< Sampling Frequency in MHz

  int mGasMixture{kXenon}; ///< Gas mixture: 0-Xe/C02 1-Ar/CO2.

 private:
  /// This is a singleton, constructor is private!
  CommonParam() = default;

  ClassDefNV(CommonParam, 1); // The constant parameters common to simulation and reconstruction
};
} // namespace trd
} // namespace o2
#endif
