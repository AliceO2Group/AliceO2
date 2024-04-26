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

#ifndef ALICEO2_MCCOMPLABEL_H
#define ALICEO2_MCCOMPLABEL_H

#include <cstdint>
#include "GPUCommonRtypes.h"
#include <string>

namespace o2
{
// Composed Label to encode MC track id, event it comes from and the source (file)

class MCCompLabel
{
 private:
  static constexpr uint64_t ul0x1 = 0x1;
  static constexpr uint64_t NotSet = 0xffffffffffffffff;
  static constexpr uint64_t Noise = 0xfffffffffffffffe;
  static constexpr uint64_t Fake = ul0x1 << 63;
  static constexpr int NReservedBits = 1;

  uint64_t mLabel = NotSet; ///< MC label encoding MCtrack ID and MCevent origin

 public:
  // number of bits reserved for MC track ID, DON'T modify this, since the
  // track ID might be negative
  static constexpr int nbitsTrackID = 31; // number of bits reserved for MC track ID
  static constexpr int nbitsEvID = 19;    // number of bits reserved for MC event ID
  static constexpr int nbitsSrcID = 8;    // number of bits reserved for MC source ID
  // the rest of the bits is reserved at the moment

  // check if the fields are defined consistently
  static_assert(nbitsTrackID + nbitsEvID + nbitsSrcID <= sizeof(uint64_t) * 8 - NReservedBits,
                "Fields cannot be stored in 64 bits");

  // mask to extract MC track ID
  static constexpr uint64_t maskTrackID = (ul0x1 << nbitsTrackID) - 1;
  // mask to extract MC track ID
  static constexpr uint64_t maskEvID = (ul0x1 << nbitsEvID) - 1;
  // mask to extract MC track ID
  static constexpr uint64_t maskSrcID = (ul0x1 << nbitsSrcID) - 1;
  // mask for all used fields
  static constexpr uint64_t maskFull = (ul0x1 << (nbitsTrackID + nbitsEvID + nbitsSrcID)) - 1;

  MCCompLabel(int trackID, int evID, int srcID, bool fake = false) { set(trackID, evID, srcID, fake); }
  MCCompLabel(bool noise = false)
  {
    if (noise) {
      mLabel = Noise;
    } else {
      mLabel = NotSet;
    }
  }
  ~MCCompLabel() = default;

  // check if label was assigned
  bool isSet() const { return mLabel != NotSet; }
  // check if label was not assigned
  bool isEmpty() const { return mLabel == NotSet; }
  // check if label comes from QED contrib
  bool isQED() const { return getSourceID() == 99; }
  // check if label corresponds to real particle (for the moment QED is not included)
  bool isNoise() const { return mLabel == Noise || isQED(); }
  // check if label was assigned as for correctly identified particle
  bool isValid() const { return isSet() && !isNoise(); }

  // check if label was assigned as for incorrectly identified particle or not set or noise
  bool isFake() const { return mLabel & Fake; }
  // check if label was assigned as for correctly identified particle
  bool isCorrect() const { return !isFake(); }

  // return 1 if the tracks are the same and correctly identified
  // 0 if the tracks are the same but at least one of them is fake
  // -1 otherwhise
  int compare(const MCCompLabel& other) const
  {
    if (getEventID() != other.getEventID() || getSourceID() != other.getSourceID()) {
      return -1;
    }
    int tr1 = getTrackID(), tr2 = other.getTrackID();
    return (tr1 == tr2) ? ((isCorrect() && other.isCorrect()) ? 1 : 0) : -1;
  }

  // allow to retrieve bare label
  uint64_t getRawValue() const { return mLabel; }

  // comparison operator, compares only label, not eventual weight or correctness info
  bool operator==(const MCCompLabel& other) const { return (mLabel & maskFull) == (other.mLabel & maskFull); }
  bool operator!=(const MCCompLabel& other) const { return (mLabel & maskFull) != (other.mLabel & maskFull); }
  // relation operators needed for some sorting methods
  bool operator<(const MCCompLabel& other) const { return (mLabel & maskFull) < (other.mLabel & maskFull); }
  bool operator>(const MCCompLabel& other) const { return (mLabel & maskFull) > (other.mLabel & maskFull); }

  // invalidate
  void unset() { mLabel = NotSet; }
  void setNoise() { mLabel = Noise; }
  void setFakeFlag(bool v = true)
  {
    if (v) {
      mLabel |= Fake;
    } else {
      mLabel &= ~Fake;
    }
  }

  void set(unsigned int trackID, int evID, int srcID, bool fake)
  {
    /// compose label: the track 1st cast to UInt32_t to preserve the sign!
    mLabel = (maskTrackID & static_cast<uint64_t>(trackID)) |
             (maskEvID & static_cast<uint64_t>(evID)) << nbitsTrackID |
             (maskSrcID & static_cast<uint64_t>(srcID)) << (nbitsTrackID + nbitsEvID);
    if (fake) {
      setFakeFlag();
    }
  }

  int getTrackID() const { return static_cast<int>(mLabel & maskTrackID); }
  int getTrackIDSigned() const { return isFake() ? -getTrackID() : getTrackID(); }
  int getEventID() const { return (mLabel >> nbitsTrackID) & maskEvID; }
  int getSourceID() const { return (mLabel >> (nbitsTrackID + nbitsEvID)) & maskSrcID; }
  uint64_t getTrackEventSourceID() const { return static_cast<uint64_t>(mLabel & maskFull); }
  void get(int& trackID, int& evID, int& srcID, bool& fake)
  {
    /// parse label
    trackID = getTrackID();
    evID = getEventID();
    srcID = getSourceID();
    fake = isFake();
  }

  void print() const;
  std::string asString() const;

  static constexpr int maxSourceID() { return maskSrcID; }
  static constexpr int maxEventID() { return maskEvID; }
  static constexpr int maxTrackID() { return maskTrackID; }
  ClassDefNV(MCCompLabel, 1);
};

std::ostream& operator<<(std::ostream& os, MCCompLabel const& c);

} // namespace o2

namespace std
{
// defining std::hash for MCCompLabel in order to be used with unordered_maps
template <>
struct hash<o2::MCCompLabel> {
 public:
  size_t operator()(o2::MCCompLabel const& label) const
  {
    return label.getRawValue();
  }
};
} // namespace std

#endif
