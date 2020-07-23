// Copyright CERN and copyright holders of ALICE O2. This software is
// // distributed under the terms of the GNU General Public License v3 (GPL
// // Version 3), copied verbatim in the file "COPYING".
// //
// // See http://alice-o2.web.cern.ch/license for full licensing information.
// //
// // In applying this license CERN does not waive the privileges and immunities
// // granted to it by virtue of its status as an Intergovernmental Organization
// // or submit itself to any jurisdiction.
//
// /// \file Tracklet2Raw.h
// /// \brief converts tracklets raw stream to raw data format
// // murrays@cern.ch
//
//
#ifndef ALICEO2_TRD_TRACKLETS2RAW_H
#define ALICEO2_TRD_TRACKLETS2RAW_H

#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/TrackletRawData.h"

namespace o2
{
namespace trd
{

class Tracklets2Raw
{

  static constexpr int NChannels;
  static constexpr int NLinksPerCRU = 30;
  static constexpr int NCRU = 36;

 public:
  Tracklets2Raw() = default;
  Tracklets2Raw(const std::string& outDir, const std::string& fileRawTrackletsName);
  void readTracklets(const std::string& outDir, const std::string& fileRawTrackletsName);

  void converTracklets(o2::trd::TrackletRaw rawTracklets, gsl::span < 2 ::trd::TriggerRecord const& mTriggerRecord);

 private:
  const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  o2::raw::RawFileWriter mWriter("TRD");
  bool mOutputPerLink = true;
  uint32_t mLinkId = 0;
  uint32_t mCRUId = 0;
  uint32_t mEndPointId = 0;
  uint32_t mFeeId = 0;
};
} // namespace trd
} // namespace o2
#endif
