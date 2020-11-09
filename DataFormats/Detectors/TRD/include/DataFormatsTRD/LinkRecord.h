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

  struct LinkId {
    union {
      uint16_t word;
      struct {
        uint16_t spare : 4;
        uint16_t side : 1;
        uint16_t layer : 3;
        uint16_t stack : 3;
        uint16_t supermodule : 5;
      };
    };
  };

 public:
  LinkRecord() = default;
  LinkRecord(const uint32_t linkid, int firstentry, int nentries) : mDataRange(firstentry, nentries) { mLinkId = linkid; }
  // LinkRecord(const LinkRecord::LinkId linkid, int firstentry, int nentries) : mDataRange(firstentry, nentries) {mLinkId.word=linkid.word;}
  LinkRecord(uint32_t sector, int stack, int layer, int side, int firstentry, int nentries) : mDataRange(firstentry, nentries) { setLinkId(sector, stack, layer, side); }

  ~LinkRecord() = default;

  void setLinkId(const uint32_t linkid) { mLinkId = linkid; }
  //  void setLinkId(const LinkId linkid) { mLinkId = linkid; }
  void setLinkId(const uint32_t sector, const uint32_t stack, const uint32_t layer, const uint32_t side);
  void setDataRange(int firstentry, int nentries) { mDataRange.set(firstentry, nentries); }
  void setIndexFirstObject(int firstentry) { mDataRange.setFirstEntry(firstentry); }
  void setNumberOfObjects(int nentries) { mDataRange.setEntries(nentries); }

  const uint32_t getLinkId() { return mLinkId; }
  //TODO come backwith a ccdb lookup.  const uint32_t getLinkHCID() { return mLinkId & 0x7ff; } // the last 11 bits.
  const uint32_t getSector() { return (mLinkId & 0xf800) >> 11; }
  const uint32_t getStack() { return (mLinkId & 0x700) >> 8; }
  const uint32_t getLayer() { return (mLinkId & 0xe0) >> 5; }
  const uint32_t getSide() { return (mLinkId & 0x10) >> 4; }
  int getNumberOfObjects() const { return mDataRange.getEntries(); }
  int getFirstEntry() const { return mDataRange.getFirstEntry(); }
  static uint32_t getHalfChamberLinkId(uint32_t detector, uint32_t rob);
  static uint32_t getHalfChamberLinkId(uint32_t sector, uint32_t stack, uint32_t layer, uint32_t side);

  void printStream(std::ostream& stream);

 private:
  uint16_t mLinkId;
  DataRange mDataRange; /// Index of the triggering event (event index and first entry in the container)
  ClassDefNV(LinkRecord, 3);
};

std::ostream& operator<<(std::ostream& stream, LinkRecord& trg);

extern void buildTrackletHCHeader(TrackletHCHeader& header, int sector, int stack, int layer, int side, int chipclock, int format);
extern void buildTrakcletHCHeader(TrackletHCHeader& header, int detector, int rob, int chipclock, int format);
} // namespace trd

} // namespace o2

#endif
//extern void buildTrackletHCHeader(TrackletHCHeader& header, int sector, int stack, int layer, int side, int chipclock, int format)
//extern void buildTrakcletlHCHeader(TrackletHCHeader& header, int detector, int rob, int chipclock, int format)
