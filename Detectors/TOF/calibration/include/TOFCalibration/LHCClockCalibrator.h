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

#ifndef TOF_LHCPHASE_CALIBRATION_H_
#define TOF_LHCPHASE_CALIBRATION_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "TOFBase/CalibTOFapi.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "CommonUtils/NameConf.h"
#include "TOFBase/Geo.h"
#include "CCDB/CcdbObjectInfo.h"
#include <array>

namespace o2
{
namespace tof
{

struct LHCClockDataHisto {
  float range = o2::tof::Geo::BC_TIME_INPS * 0.5;
  int nbins = 1000;
  float v2Bin = nbins / (2 * range);
  int entries = 0;
  o2::tof::CalibTOFapi* calibApi;
  std::vector<float> histo{0};

  LHCClockDataHisto();

  LHCClockDataHisto(int nb, float r, o2::tof::CalibTOFapi* api) : nbins(nb), range(r), v2Bin(0), calibApi(api)
  {
    if (r <= 0. || nb < 1) {
      throw std::runtime_error("Wrong initialization of the histogram");
    }
    v2Bin = nbins / (2 * range);
    histo.resize(nbins, 0.);
  }

  size_t getEntries() const { return entries; }
  void print() const;
  void fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data);
  void merge(const LHCClockDataHisto* prev);

  ClassDefNV(LHCClockDataHisto, 1);
};

class LHCClockCalibrator final : public o2::calibration::TimeSlotCalibration<o2::dataformats::CalibInfoTOF, o2::tof::LHCClockDataHisto>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::tof::LHCClockDataHisto>;
  using CalibTOFapi = o2::tof::CalibTOFapi;
  using LHCphase = o2::dataformats::CalibLHCphaseTOF;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;
  using LHCphaseVector = std::vector<LHCphase>;

 public:
  LHCClockCalibrator(int minEnt = 500, int nb = 10000, float r = 244000, const std::string path = o2::base::NameConf::getCCDBServer()) : mMinEntries(minEnt), mNBins(nb), mRange(r) { mCalibTOFapi->setURL(path); }
  ~LHCClockCalibrator() final = default;
  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->entries >= mMinEntries; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  const LHCphaseVector& getLHCphaseVector() const { return mLHCphaseVector; }
  const CcdbObjectInfoVector& getLHCphaseInfoVector() const { return mInfoVector; }
  CcdbObjectInfoVector& getLHCphaseInfoVector() { return mInfoVector; }

  void setCalibTOFapi(CalibTOFapi* api) { mCalibTOFapi = api; }
  CalibTOFapi* getCalibTOFapi() const { return mCalibTOFapi; }

 private:
  int mMinEntries = 0;
  int mNBins = 0;
  float mRange = 0.;
  CalibTOFapi* mCalibTOFapi = nullptr;
  CcdbObjectInfoVector mInfoVector; // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying LHCPhase
  LHCphaseVector mLHCphaseVector;   // vector of LhcPhase, each element is filled in "process" when we finalize one slot (multiple can be finalized during the same "process", which is why we have a vector. Each element is to be considered the output of the device, and will go to the CCDB

  ClassDefOverride(LHCClockCalibrator, 1);
};

} // end namespace tof
} // end namespace o2

#endif /* TOF_LHCPHASE_CALIBRATION_H_ */
