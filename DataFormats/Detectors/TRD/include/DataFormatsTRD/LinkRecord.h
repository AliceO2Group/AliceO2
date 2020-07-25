// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_LINKRECORD_H
#define ALICEO2_TRD_LINKRECORD_H

#include <iosfwd>
#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsTRD/RawData.h"

namespace o2
{

namespace trd
{

/// \class LinkRecord
/// \brief Header for data corresponding to the indexing of the links in the raw data output
/// adapted from DataFormatsTRD/TriggerRecord
class LinkRecord
{
  using DataRange = o2::dataformats::RangeReference<int>;

 public:
  LinkRecord() = default;
  LinkRecord(const uint32_t hcid, int firstentry, int nentries) : mLinkId(hcid), mDataRange(firstentry, nentries) {}
  ~LinkRecord() = default;

  void setLinkId(const uint32_t linkid) { mLinkId = linkid; }
  void setDataRange(int firstentry, int nentries) { mDataRange.set(firstentry, nentries); }
  void setIndexFirstObject(int firstentry) { mDataRange.setFirstEntry(firstentry); }
  void setNumberOfObjects(int nentries) { mDataRange.setEntries(nentries); }

  uint32_t getLinkId() { return mLinkId; }
  int getNumberOfObjects() const { return mDataRange.getEntries(); }
  int getFirstEntry() const { return mDataRange.getFirstEntry(); }

  void printStream(std::ostream& stream) const;

 private:
  uint32_t mLinkId;     /// The link ID for this set of data, hcid as well
  DataRange mDataRange; /// Index of the triggering event (event index and first entry in the container)

  ClassDefNV(LinkRecord, 1);
};

std::ostream& operator<<(std::ostream& stream, const LinkRecord& trg);

} // namespace trd

} // namespace o2

#endif
