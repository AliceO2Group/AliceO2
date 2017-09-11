// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MCCOMPLABEL_H
#define ALICEO2_MCCOMPLABEL_H

#include <Rtypes.h>

namespace o2
{
// Composed Label to encode MC track id, event it comes from and the source (file)

class MCCompLabel
{
 private:
  static constexpr ULong64_t ul0x1 = 0x1;
  static constexpr ULong64_t NotSet = 0xffffffffffffffff;

  ULong64_t mLabel = NotSet; ///< MC label encoding MCtrack ID and MCevent origin

  void checkFieldConsistensy();
  
 public:

  // number of bits reserved for MC track ID, DON'T modify this, sine the
  // track ID might be negative
  static constexpr int nbitsTrackID = sizeof(int)*8;
  static constexpr int nbitsEvID = 19;    // number of bits reserved for MC event ID
  static constexpr int nbitsSrcID = 8;    // number of bits reserved for MC source ID
  // the rest of the bits is reserved at the moment

  // mask to extract MC track ID
  static constexpr ULong64_t maskTrackID = (ul0x1 << nbitsTrackID) - 1;
  // mask to extract MC track ID
  static constexpr ULong64_t maskEvID = (ul0x1 << nbitsEvID) - 1;
  // mask to extract MC track ID
  static constexpr ULong64_t maskSrcID = (ul0x1 << nbitsSrcID) - 1;
  // mask for all used fields
  static constexpr ULong64_t maskFull = (ul0x1 << (nbitsTrackID + nbitsEvID + nbitsSrcID)) - 1;

  MCCompLabel(int trackID, int evID = 0, int srcID = 0) { set(trackID, evID, srcID); }
  MCCompLabel() = default;
  ~MCCompLabel() = default;

  // check if label was assigned
  bool isSet() const { return mLabel != NotSet; }

  // check if label was not assigned
  bool isEmpty() const { return mLabel == NotSet; }

  // check if label was assigned with non-negaive trackID
  bool isPosTrackID() const { return getTrackID() >= 0; }
  
  // conversion op-r
  operator ULong64_t() const { return mLabel; }
  // invalidate
  void unset() { mLabel = NotSet; }
  void set(int trackID, int evID, int srcID)
  {
    /// compose label: the track 1st cast to UInt32_t to preserve the sign!
    mLabel = (maskTrackID & static_cast<ULong64_t>( static_cast<UInt_t>(trackID))) |
             (maskEvID & static_cast<ULong64_t>(evID)) << nbitsTrackID |
             (maskSrcID & static_cast<ULong64_t>(srcID)) << (nbitsTrackID + nbitsEvID);
  }
  
  int getTrackID() const { return static_cast<int>(mLabel & maskTrackID); }
  int getEventID() const { return (mLabel >> nbitsTrackID) & maskEvID; }
  int getSourceID() const { return (mLabel >> (nbitsTrackID + nbitsEvID)) & maskSrcID; }
  void get(int& trackID, int& evID, int& srcID)
  {
    /// parse label
    trackID = getTrackID();
    evID = getEventID();
    srcID = getSourceID();
  }

  void print() const;

  static constexpr int maxSourceID()  { return maskSrcID; }
  static constexpr int maxEventID()   { return maskEvID; }
  static constexpr int maxTrackID()   { return maskTrackID; }
  
  ClassDefNV(MCCompLabel, 1);
};
}

std::ostream& operator<<(std::ostream& os, const o2::MCCompLabel& c);

#endif
