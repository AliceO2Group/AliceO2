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

#ifndef O2_TRD_DIFFANDTIMESTRUCTESTIMATOR_H
#define O2_TRD_DIFFANDTIMESTRUCTESTIMATOR_H

#include <array>
#include "DataFormatsTRD/Constants.h"
#include "TRDSimulation/SimParam.h"
#include "TRDSimulation/Garfield.h"

namespace o2
{
namespace trd
{

/// A class to calculate diffusion and time structure values (GARFIELD model)
/// no longer a singleton so that we can use it in a multithreaded context.
class DiffusionAndTimeStructEstimator
{
 public:
  DiffusionAndTimeStructEstimator() = delete; // upon creation gas mixture and B-field must be provided
  DiffusionAndTimeStructEstimator(SimParam::GasMixture gas, float bz) : mGasMixture(gas), mBz(bz) {}

  /// determines the diffusion coefficients as a function of drift velocity
  bool getDiffCoeff(float& dl, float& dt, float vdrift);

  /// determines drift time as function of drift velocity and coordinates
  float timeStruct(float vdrift, float xd, float z, bool* errFlag = nullptr);

 private:
  bool sampleTimeStruct(float vdrift);

  std::array<float, garfield::TIMEBINSGARFIELD * garfield::ZBINSGARFIELD> mTimeStruct1; ///< cached Time Structure of Drift Cells (for last vdrift value)
  std::array<float, garfield::TIMEBINSGARFIELD * garfield::ZBINSGARFIELD> mTimeStruct2; ///< cached Time Structure of Drift Cells (for last vdrift value)
  float mVDlo;                                                                          ///<  Lower drift velocity, for interpolation
  float mVDhi;                                                                          ///<  Higher drift velocity, for interpolation
  float mInvBinWidth;                                                                   ///<  caching 1/(mVDhi - mVDlo)
  float mTimeLastVdrift = -1.f;                                                         ///<  The structures are valid for this mLastVdrift (caching)

  // for the diffusion part
  float mDiffLastVdrift = -1.f;
  float mDiffusionL = -1.f;
  float mDiffusionT = -1.f;

  SimParam::GasMixture mGasMixture;
  float mBz;
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_DIFFANDTIMESTRUCTESTIMATOR_H
