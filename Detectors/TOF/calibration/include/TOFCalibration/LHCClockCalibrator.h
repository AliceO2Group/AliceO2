// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TOF_LHCPHASE_CALIBRATION_H_
#define TOF_LHCPHASE_CALIBRATION_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "TOFCalibration/CalibTOFapi.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
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
  void fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data);
  void merge(const LHCClockDataHisto* prev);

  ClassDefNV(LHCClockDataHisto, 1);
};

class LHCClockCalibrator : public o2::calibration::TimeSlotCalibration<o2::dataformats::CalibInfoTOF, o2::tof::LHCClockDataHisto>
{
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<o2::tof::LHCClockDataHisto>;
  using CalibTOFapi = o2::tof::CalibTOFapi;
  using LHCphase = o2::dataformats::CalibLHCphaseTOF;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;
  using LHCphaseVector = std::vector<LHCphase>;

 public:
  LHCClockCalibrator(int minEnt = 500, int nb = 1000, float r = 24400, const std::string path = "http://ccdb-test.cern.ch:8080") : mMinEntries(minEnt), mNBins(nb), mRange(r) { mCalibTOFapi.setURL(path); }
  ~LHCClockCalibrator() final = default;
  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->entries >= mMinEntries; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  const LHCphaseVector& getLHCphaseVector() const { return mLHCphaseVector; }
  const CcdbObjectInfoVector& getLHCphaseInfoVector() const { return mInfoVector; }
  CcdbObjectInfoVector& getLHCphaseInfoVector() { return mInfoVector; }

 private:
  int mMinEntries = 0;
  int mNBins = 0;
  float mRange = 0.;
  CalibTOFapi mCalibTOFapi;
  CcdbObjectInfoVector mInfoVector; // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying LHCPhase
  LHCphaseVector mLHCphaseVector;   // vector of LhcPhase, each element is filled in "process" when we finalize one slot (multiple can be finalized during the same "process", which is why we have a vector. Each element is to be considered the output of the device, and will go to the CCDB

  ClassDefOverride(LHCClockCalibrator, 1);
};

} // end namespace tof
} // end namespace o2

#endif /* TOF_LHCPHASE_CALIBRATION_H_ */
