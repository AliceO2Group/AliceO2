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
  LinkRecord(const uint32_t linkid, int firstentry, int nentries) : mDataRange(firstentry, nentries) { mLinkId = linkid; }
  // LinkRecord(const LinkRecord::LinkId linkid, int firstentry, int nentries) : mDataRange(firstentry, nentries) {mLinkId.word=linkid.word;}
  LinkRecord(uint32_t sector, int stack, int layer, int side, int firstentry, int nentries) : mDataRange(firstentry, nentries) { setLinkId(sector, stack, layer, side); }

  ~LinkRecord() = default;

  void setLinkId(const uint32_t linkid) { mLinkId = linkid; }
  void setLinkId(const uint32_t sector, const uint32_t stack, const uint32_t layer, const uint32_t side);
  void setDataRange(int firstentry, int nentries) { mDataRange.set(firstentry, nentries); }
  void setIndexFirstObject(int firstentry) { mDataRange.setFirstEntry(firstentry); }
  void setNumberOfObjects(int nentries) { mDataRange.setEntries(nentries); }
  void setSector(const int sector) { mLinkId |= ((sector << supermodulebs) & supermodulemask); }
  void setStack(const int stack) { mLinkId |= ((stack << stackbs) & stackmask); }
  void setLayer(const int layer) { mLinkId |= ((layer << layerbs) & layermask); }
  void setSide(const int side) { mLinkId |= ((side << sidebs) & sidemask); }
  void setSpare(const int spare = 0) { mLinkId |= ((spare << sparebs) & sparemask); }

  const uint32_t getLinkId() { return mLinkId; }
  //TODO come backwith a ccdb lookup.  const uint32_t getLinkHCID() { return mLinkId & 0x7ff; } // the last 11 bits.
  const uint32_t getSector() { return (mLinkId & supermodulemask) >> supermodulebs; }
  const uint32_t getStack() { return (mLinkId & stackmask) >> stackbs; }
  const uint32_t getLayer() { return (mLinkId & layermask) >> layerbs; }
  const uint32_t getSide() { return (mLinkId & sidemask) >> sidebs; }
  int getNumberOfObjects() const { return mDataRange.getEntries(); }
  int getFirstEntry() const { return mDataRange.getFirstEntry(); }
  static uint32_t getHalfChamberLinkId(uint32_t detector, uint32_t rob);
  static uint32_t getHalfChamberLinkId(uint32_t sector, uint32_t stack, uint32_t layer, uint32_t side);

  void printStream(std::ostream& stream);
  // bit masks for the above raw data;
  static constexpr uint64_t sparemask = 0x000f;
  static constexpr uint64_t sidemask = 0x0010;
  static constexpr uint64_t layermask = 0x00e0;
  static constexpr uint64_t stackmask = 0x0700;
  static constexpr uint64_t supermodulemask = 0xf800;
  //bit shifts for the above raw data
  static constexpr uint64_t sparebs = 0;
  static constexpr uint64_t sidebs = 4;
  static constexpr uint64_t layerbs = 5;
  static constexpr uint64_t stackbs = 8;
  static constexpr uint64_t supermodulebs = 11;

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
