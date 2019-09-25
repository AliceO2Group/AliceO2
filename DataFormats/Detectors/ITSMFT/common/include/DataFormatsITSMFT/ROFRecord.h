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
/// \brief Definition of the ITSMFT ROFrame (trigger) record

#ifndef ALICEO2_ITSMFT_ROFRECORD_H
#define ALICEO2_ITSMFT_ROFRECORD_H

#include "CommonDataFormat/EvIndex.h"
#include "CommonDataFormat/InteractionRecord.h"

namespace o2
{
namespace itsmft
{

/// ROFRecord class encodes the trigger interaction record of given ROF and
/// the reference on the 1st object (digit, cluster etc) of this ROF in the data tree

class ROFRecord
{
  using EvIdx = o2::dataformats::EvIndex<int, int>;
  using BCData = o2::InteractionRecord;
  using ROFtype = unsigned int;

 public:
  ROFRecord() = default;
  ROFRecord(const BCData& bc, ROFtype rof, EvIdx entry, int n)
    : mBCData(bc), mROFEntry(entry), mROFrame(rof), mNROFEntries(n) {}

  void setBCData(const BCData& bc) { mBCData = bc; }
  void setROFrame(ROFtype rof) { mROFrame = rof; }
  void setROFEntry(EvIdx entry) { mROFEntry = entry; }
  void setNROFEntries(int n) { mNROFEntries = n; }

  const BCData& getBCData() const { return mBCData; }
  BCData& getBCData() { return mBCData; }
  EvIdx getROFEntry() const { return mROFEntry; }
  EvIdx& getROFEntry() { return mROFEntry; }
  ROFtype getROFrame() const { return mROFrame; }
  int getNROFEntries() const { return mNROFEntries; }

  void clear()
  {
    mROFEntry.clear();
    mNROFEntries = 0;
    mBCData.clear();
  }
  void print() const;

 private:
  o2::InteractionRecord mBCData; // BC data for given trigger
  EvIdx mROFEntry;               //< reference on the 1st object of the ROF in data
  ROFtype mROFrame = 0;          //< frame ID
  int mNROFEntries = 0;          //< number of objects of the ROF

  ClassDefNV(ROFRecord, 1);
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
  void print() const;
  ClassDefNV(MC2ROFRecord, 1);
};
} // namespace itsmft
} // namespace o2

#endif
