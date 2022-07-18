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

/// \file VDriftITSTPCCalibration.h
/// \brief time slot calibration for VDrift via TPCITSMatches tgl difference
/// \author ruben.shahoian@cern.ch

#ifndef TPC_VDrifITSTPCCalibration_H_
#define TPC_VDrifITSTPCCalibration_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "CommonDataFormat/FlatHisto2D.h"
#include "DataFormatsTPC/VDriftCorrFact.h"
#include "CommonDataFormat/Pair.h"
#include "CCDB/CcdbObjectInfo.h"

namespace o2::tpc
{
struct TPCVDTglContainer {
  std::unique_ptr<o2::dataformats::FlatHisto2D_f> histo;
  size_t entries = 0;

  TPCVDTglContainer(int ntgl, float tglMax, int ndtgl, float dtglMax)
  {
    histo = std::make_unique<o2::dataformats::FlatHisto2D_f>(ntgl, -tglMax, tglMax, ndtgl, -dtglMax, dtglMax);
  }

  TPCVDTglContainer(const TPCVDTglContainer& src)
  {
    histo = std::make_unique<o2::dataformats::FlatHisto2D_f>(*(src.histo.get()));
    entries = src.entries;
  }

  void fill(const gsl::span<const o2::dataformats::Pair<float, float>> data)
  {
    for (auto p : data) {
      histo->fill(p.first, p.first - p.second);
    }
    entries += data.size();
  }

  void merge(const TPCVDTglContainer* other)
  {
    entries += other->entries;
    histo->add(*(other->histo));
  }

  void print() const
  {
    LOG(info) << "Nentries = " << entries;
  }
  ClassDefNV(TPCVDTglContainer, 1);
};

class TPCVDriftTglCalibration : public o2::calibration::TimeSlotCalibration<o2::dataformats::Pair<float, float>, TPCVDTglContainer>
{
  using Slot = o2::calibration::TimeSlot<TPCVDTglContainer>;

 public:
  TPCVDriftTglCalibration() = default;
  TPCVDriftTglCalibration(int ntgl, float tglMax, int ndtgl, float dtglMax, uint32_t slotL, size_t minEnt) : mNBinsTgl(ntgl), mMaxTgl(tglMax), mNBinsDTgl(ndtgl), mMaxDTgl(dtglMax), mMineEntriesPerSlot(minEnt)
  {
    setSlotLength(slotL);
    setMaxSlotsDelay(10);
  }

  ~TPCVDriftTglCalibration() final = default;

  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->entries >= mMineEntriesPerSlot; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  const std::vector<o2::tpc::VDriftCorrFact>& getVDPerSlot() const { return mVDPerSlot; }
  const std::vector<o2::ccdb::CcdbObjectInfo>& getCCDBInfoPerSlot() const { return mCCDBInfoPerSlot; }
  std::vector<o2::tpc::VDriftCorrFact>& getVDPerSlot() { return mVDPerSlot; }
  std::vector<o2::ccdb::CcdbObjectInfo>& getCCDBInfoPerSlot() { return mCCDBInfoPerSlot; }

  void setSaveHistosFile(const std::string& f) { mSaveHistosFile = f; }

 private:
  size_t mMineEntriesPerSlot = 10000;
  int mNBinsTgl = 10;
  int mNBinsDTgl = 100;
  float mMaxTgl = 1.;
  float mMaxDTgl = 0.2;
  std::string mSaveHistosFile{};
  std::vector<o2::tpc::VDriftCorrFact> mVDPerSlot;
  std::vector<o2::ccdb::CcdbObjectInfo> mCCDBInfoPerSlot;

  ClassDefNV(TPCVDriftTglCalibration, 1);
};

} // namespace o2::tpc
#endif
