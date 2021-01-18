// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

 public:
  ROFRecord() = default;
  ROFRecord(const BCData& bc, int firstIdx, int nEntries) : mBCData(bc), mDataRef(firstIdx, nEntries) {}

  /// get the interaction record
  const BCData& getBCData() const { return mBCData; }
  /// get the interaction record
  BCData& getBCData() { return mBCData; }
  /// set the interaction record
  void setBCData(const BCData& bc) { mBCData = bc; }

  /// get the number of associated objects
  int getNEntries() const { return mDataRef.getEntries(); }
  /// get the index of the first associated object
  int getFirstIdx() const { return mDataRef.getFirstEntry(); }
  /// get the index of the last associated object
  int getLastIdx() const { return mDataRef.getFirstEntry() + mDataRef.getEntries() - 1; }
  /// set the number of associated objects and the index of the first one
  void setDataRef(int firstIdx, int nEntries) { mDataRef.set(firstIdx, nEntries); }

 private:
  BCData mBCData{};   ///< interaction record
  DataRef mDataRef{}; ///< reference to the associated objects

  ClassDefNV(ROFRecord, 1);
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_ROFRECORD_H
