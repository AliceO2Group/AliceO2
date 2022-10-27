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

/// \file CalibratorVdExB.h
/// \brief TimeSlot-based calibration of vDrift and ExB
/// \author Ole Schmidt

#ifndef O2_TRD_CALIBRATORVDEXB_H
#define O2_TRD_CALIBRATORVDEXB_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/AngularResidHistos.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsTRD/CalVdriftExB.h"

#include "Rtypes.h"
#include "TProfile.h"

#include <array>
#include <cstdlib>

namespace o2
{
namespace trd
{

struct FitFunctor {
  double operator()(const double* par) const;
  double calculateDeltaAlphaSim(double vdFit, double laFit, double impactAng) const;
  std::array<std::unique_ptr<TProfile>, constants::MAXCHAMBER> profiles; ///< profile histograms for each TRD chamber
  double vdPreCorr;                                                      // TODO: these values should eventually be taken from CCDB
  double laPreCorr;                                                      // TODO: these values should eventually be taken from CCDB
  int currDet;                                                           ///< the current TRD chamber number
  float lowerBoundAngleFit;
  float upperBoundAngleFit;
};

class CalibratorVdExB final : public o2::calibration::TimeSlotCalibration<o2::trd::AngularResidHistos, o2::trd::AngularResidHistos>
{
  using Slot = o2::calibration::TimeSlot<o2::trd::AngularResidHistos>;

 public:
  enum ParamIndex : int {
    LA,
    VD
  };
  CalibratorVdExB(size_t nMin = 40'000, bool enableOut = false) : mMinEntries(nMin), mEnableOutput(enableOut) {}
  ~CalibratorVdExB() final = default;

  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->getNEntries() >= mMinEntries; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tStart, TFType tEnd) final;

  const std::vector<o2::trd::CalVdriftExB>& getCcdbObjectVector() const { return mObjectVector; }
  std::vector<o2::ccdb::CcdbObjectInfo>& getCcdbObjectInfoVector() { return mInfoVector; }

  void initProcessing();

 private:
  bool mInitDone{false}; ///< flag to avoid creating the TProfiles multiple times
  size_t mMinEntries; ///< minimum total number of angular deviations (on average ~3 entries per bin for each TRD chamber)
  bool mEnableOutput; //< enable output of calibration fits and tprofiles in a root file instead of the ccdb
  FitFunctor mFitFunctor; ///< used for minimization procedure
  std::vector<o2::ccdb::CcdbObjectInfo> mInfoVector; ///< vector of CCDB infos; each element is filled with CCDB description of accompanying CCDB calibration object
  std::vector<o2::trd::CalVdriftExB> mObjectVector;  ///< vector of CCDB calibration objects; the extracted vDrift and ExB per chamber for given slot
  ClassDefOverride(CalibratorVdExB, 2);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CALIBRATORVDEXB_H
