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
/// \brief Definition of the ITSMFT ROFrame (trigger) record

#ifndef ALICEO2_ITSMFT_ROFRECORD_H
#define ALICEO2_ITSMFT_ROFRECORD_H

#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <gsl/span>

namespace o2
{
namespace itsmft
{

/// ROFRecord class encodes the trigger interaction record of given ROF and
/// the reference on the 1st object (digit, cluster etc) of this ROF in the data tree

class ROFRecord
{

 public:
  using EvIdx = o2::dataformats::RangeReference<int, int>;
  using BCData = o2::InteractionRecord;
  using ROFtype = unsigned int;

  ROFRecord() = default;
  ROFRecord(const BCData& bc, ROFtype rof, int idx, int n)
    : mBCData(bc), mROFEntry(idx, n), mROFrame(rof) {}

  void setBCData(const BCData& bc) { mBCData = bc; }
  void setROFrame(ROFtype rof) { mROFrame = rof; }
  void setEntry(EvIdx entry) { mROFEntry = entry; }
  void setFirstEntry(int idx) { mROFEntry.setFirstEntry(idx); }
  void setNEntries(int n) { mROFEntry.setEntries(n); }

  uint32_t getFlags() const { return mBits; }
  void setFlags(uint32_t flags) { mBits = flags; }
  void setFlag(uint8_t flagIndex) { mBits |= (1 << flagIndex); }
  void resetFlag(uint8_t flagIndex) { mBits &= ~(1 << flagIndex); }
  bool getFlag(uint8_t flagIndex) const { return mBits & (1 << flagIndex); }
  void clearAllFlags() { mBits = 0; }
  void setAllFlags() { mBits = ~0; }

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }
  EvIdx getEntry() const { return mROFEntry; }
  EvIdx& getEntry() { return mROFEntry; }
  int getNEntries() const { return mROFEntry.getEntries(); }
  int getFirstEntry() const { return mROFEntry.getFirstEntry(); }
  ROFtype getROFrame() const { return mROFrame; }

  void clear()
  {
    mROFEntry.clear();
    mBCData.clear();
  }

  template <typename T>
  gsl::span<const T> getROFData(const gsl::span<const T> tfdata) const
  {
    return getNEntries() ? gsl::span<const T>(&tfdata[getFirstEntry()], getNEntries()) : gsl::span<const T>();
  }

  template <typename T>
  const T* getROFDataAt(int i, const gsl::span<const T> tfdata) const
  {
    return i < getNEntries() ? &tfdata[getFirstEntry() + i] : nullptr;
  }

  template <typename T>
  gsl::span<const T> getROFData(const std::vector<T>& tfdata) const
  {
    return getNEntries() ? gsl::span<const T>(&tfdata[getFirstEntry()], getNEntries()) : gsl::span<const T>();
  }

  template <typename T>
  const T* getROFDataAt(int i, const std::vector<T>& tfdata) const
  {
    return i < getNEntries() ? &tfdata[getFirstEntry() + i] : nullptr;
  }

  std::string asString() const;
  void print() const;
  friend std::ostream& operator<<(std::ostream& output, const ROFRecord& rec);

 private:
  o2::InteractionRecord mBCData; // BC data for given trigger
  EvIdx mROFEntry;               //< reference on the 1st object of the ROF in data
  ROFtype mROFrame = 0;          //< frame ID
  uint32_t mBits = 0;
  ClassDefNV(ROFRecord, 3);
};

/// this is a simple reference connecting (composed) MC event ID (from the EventRecord of the RunContext)
/// with the entry in the ROFrecords entry
struct MC2ROFRecord {
  using ROFtype = unsigned int;

  int eventRecordID = -1; ///< MCevent entry in the EventRecord
  int rofRecordID = 0;    ///< 1st entry in the ROFRecords vector
  ROFtype minROF = 0;     ///< 1st ROFrame it contributed
  ROFtype maxROF = 0;     ///< last ROF event contributed

  MC2ROFRecord() = default;
  MC2ROFRecord(int evID, int rofRecID, ROFtype mnrof, ROFtype mxrof) : eventRecordID(evID), rofRecordID(rofRecID), minROF(mnrof), maxROF(mxrof) {}
  int getNROFs() const { return (rofRecordID < 0 || minROF > maxROF) ? 0 : (maxROF - minROF); }
  std::string asString() const;
  void print() const;
  friend std::ostream& operator<<(std::ostream& output, const MC2ROFRecord& rec);

  ClassDefNV(MC2ROFRecord, 1);
};
} // namespace itsmft
} // namespace o2

#endif
