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

/// \file ROFRecord.h
/// \brief Definition of the MCH ROFrame record
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_ROFRECORD_H
#define ALICEO2_MCH_ROFRECORD_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/TimeStamp.h"

#include <iosfwd>
#include <utility>

namespace o2
{
namespace mch
{

/// ROFRecord class encodes the trigger interaction record of a given ROF and
/// the location of the associated objects (digit, cluster, etc) in the data container
class ROFRecord
{
  using BCData = o2::InteractionRecord;
  using DataRef = o2::dataformats::RangeReference<int, int>;
  using Time = o2::dataformats::TimeStampWithError<float, float>;

 public:
  ROFRecord() = default;
  ROFRecord(const BCData& bc, int firstIdx, int nEntries) : mBCData(bc), mDataRef(firstIdx, nEntries) {}
  ROFRecord(const BCData& bc, int firstIdx, int nEntries, int bcWidth) : mBCData(bc), mDataRef(firstIdx, nEntries), mBCWidth(bcWidth) {}

  /// get the interaction record
  const BCData& getBCData() const { return mBCData; }
  /// get the interaction record
  BCData& getBCData() { return mBCData; }
  /// set the interaction record
  void setBCData(const BCData& bc) { mBCData = bc; }

  std::pair<Time, bool> getTimeMUS(const BCData& startIR, uint32_t nOrbits = 128, bool printError = false) const;

  /// get the number of associated objects
  int getNEntries() const { return mDataRef.getEntries(); }
  /// get the index of the first associated object
  int getFirstIdx() const { return mDataRef.getFirstEntry(); }
  /// get the index of the last associated object
  int getLastIdx() const { return mDataRef.getFirstEntry() + mDataRef.getEntries() - 1; }
  /// set the number of associated objects and the index of the first one
  void setDataRef(int firstIdx, int nEntries) { mDataRef.set(firstIdx, nEntries); }

  /// get the time span by this ROF, in BC unit
  int getBCWidth() const { return mBCWidth; }

  bool operator==(const ROFRecord& other) const
  {
    return mBCData == other.mBCData &&
           mDataRef == other.mDataRef &&
           mBCWidth == other.mBCWidth;
  }
  bool operator!=(const ROFRecord& other) const { return !(*this == other); }
  bool operator<(const ROFRecord& other) const
  {
    if (mBCData == other.mBCData) {
      if (mBCWidth == other.mBCWidth) {
        return mDataRef.getFirstEntry() < other.mDataRef.getFirstEntry();
      } else {
        return mBCWidth < other.mBCWidth;
      }
    }
    return mBCData < other.mBCData;
  }

 private:
  BCData mBCData{};   ///< interaction record
  DataRef mDataRef{}; ///< reference to the associated objects
  int mBCWidth{4};    ///< time span of this ROF

  ClassDefNV(ROFRecord, 2);
};

std::ostream& operator<<(std::ostream& os, const ROFRecord& rof);

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_ROFRECORD_H
