// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATAFORMATSTRK_ROFRECORD_H
#define ALICEO2_DATAFORMATSTRK_ROFRECORD_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include <Rtypes.h>
#include <cstdint>
#include <string>

namespace o2::trk
{

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

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }
  EvIdx getEntry() const { return mROFEntry; }
  EvIdx& getEntry() { return mROFEntry; }
  int getNEntries() const { return mROFEntry.getEntries(); }
  int getFirstEntry() const { return mROFEntry.getFirstEntry(); }
  ROFtype getROFrame() const { return mROFrame; }

  std::string asString() const;

 private:
  o2::InteractionRecord mBCData;
  EvIdx mROFEntry;
  ROFtype mROFrame = 0;

  ClassDefNV(ROFRecord, 1);
};

struct MC2ROFRecord {
  using ROFtype = unsigned int;

  int eventRecordID = -1;
  int rofRecordID = 0;
  ROFtype minROF = 0;
  ROFtype maxROF = 0;

  MC2ROFRecord() = default;
  MC2ROFRecord(int evID, int rofRecID, ROFtype mnrof, ROFtype mxrof) : eventRecordID(evID), rofRecordID(rofRecID), minROF(mnrof), maxROF(mxrof) {}

  ClassDefNV(MC2ROFRecord, 1);
};

} // namespace o2::trk

#endif
