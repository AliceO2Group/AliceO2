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

#ifndef FT0_LHCPHASE_CALIBRATION_H_
#define FT0_LHCPHASE_CALIBRATION_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "DataFormatsFT0/LHCphaseCalibrationObject.h"
#include "CommonConstants/LHCConstants.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/CcdbObjectInfo.h"
#include <array>

namespace o2
{
namespace ft0
{

struct LHCClockDataHisto {
  float range = 1000. * o2::constants::lhc::LHCBunchSpacingNS * 0.5; // BC in PS
  int nbins = 1000;
  float v2Bin = nbins / (2 * range);
  int entries = 0;
  std::vector<float> histo{0};

  LHCClockDataHisto();

  LHCClockDataHisto(int nb, float r) : nbins(nb), range(r), v2Bin(0)
  {
    if (r <= 0. || nb < 1) {
      throw std::runtime_error("Wrong initialization of the histogram");
    }
    v2Bin = nbins / (2 * range);
    histo.resize(nbins, 0.);
  }

  size_t getEntries() const { return entries; }
  void print() const;
  void fill(const gsl::span<const FT0CalibrationInfoObject>& data);
  void merge(const LHCClockDataHisto* prev);

  ClassDefNV(LHCClockDataHisto, 1);
};

class LHCClockCalibrator
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::ft0::LHCClockDataHisto>;
  using LHCphase = o2::ft0::LHCphaseCalibrationObject;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;
  using LHCphaseVector = std::vector<LHCphase>;

 public:
  LHCClockCalibrator(int minEnt = 500, int nb = 1000, float r = 24400, const std::string path = o2::base::NameConf::getCCDBServer()) : mMinEntries(minEnt), mNBins(nb), mRange(r) {}
  ~LHCClockCalibrator() = default;
  bool hasEnoughData(const Slot& slot) const { return slot.getContainer()->entries >= mMinEntries; }
  void initOutput();
  void finalizeSlot(Slot& slot);
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend);

  const LHCphaseVector& getLHCphaseVector() const { return mLHCphaseVector; }
  const CcdbObjectInfoVector& getLHCphaseInfoVector() const { return mInfoVector; }
  CcdbObjectInfoVector& getLHCphaseInfoVector() { return mInfoVector; }

 private:
  int mMinEntries = 0;
  int mNBins = 0;
  float mRange = 0.;
  CcdbObjectInfoVector mInfoVector; // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying LHCPhase
  LHCphaseVector mLHCphaseVector;   // vector of LhcPhase, each element is filled in "process" when we finalize one slot (multiple can be finalized during the same "process", which is why we have a vector. Each element is to be considered the output of the device, and will go to the CCDB

  ClassDef(LHCClockCalibrator, 1);
};

} // end namespace ft0
} // end namespace o2

#endif /* FT0_LHCPHASE_CALIBRATION_H_ */
