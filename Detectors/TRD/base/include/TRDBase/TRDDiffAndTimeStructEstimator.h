// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDDIFFANDTIMESTRUCTESTIMATOR_H
#define O2_TRDDIFFANDTIMESTRUCTESTIMATOR_H

#include <array>

namespace o2
{
namespace trd
{

// CONTANT TIME STRUCTURE DATA FROM GARFIELD
constexpr int ktimebin = 38;
constexpr int kZbin = 11;

/// A class to calculate diffusion and time structure values (GARFIELD model)
/// (used in digitization). Class was split off trom TRDCommonParam
/// and is no longer a singleton so that we can use it in a multithreaded context.
class TRDDiffusionAndTimeStructEstimator
{
 public:
  TRDDiffusionAndTimeStructEstimator() = default;

  /// determines the diffusion coefficients as a function of drift velocity
  bool GetDiffCoeff(float& dl, float& dt, float vdrift);

  /// determines drift time as function of drift velocity and coordinates
  float TimeStruct(float vdrift, float xd, float z);

 private:
  void SampleTimeStruct(float vdrift);

  std::array<float, ktimebin * kZbin> mTimeStruct1; //! cached Time Structure of Drift Cells (for last vdrift value)
  std::array<float, ktimebin * kZbin> mTimeStruct2; //! cached Time Structure of Drift Cells (for last vdrift value)
  float mVDlo;                                      //!  Lower drift velocity, for interpolation
  float mVDhi;                                      //!  Higher drift velocity, for interpolation
  float mInvBinWidth;                               //!  caching 1/(mVDhi - mVDlo)
  float mTimeLastVdrift = -1.f;                     //!  The structures are valid for this mLastVdrift (caching)

  // for the diffusion part
  float mDiffLastVdrift = -1.f;
  float mDiffusionL = -1.f;
  float mDiffusionT = -1.f;
};

} // namespace trd
} // namespace o2

#endif //O2_TRDDIFFANDTIMESTRUCTESTIMATOR_H
