// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MCVEVENTLABEL_H
#define ALICEO2_MCVEVENTLABEL_H

#include "GPUCommonRtypes.h"
#include <cmath>
#include <cassert>

namespace o2
{
// Composed Label to encode MC event and the source (file) + weight

class MCEventLabel
{
 private:
  static constexpr uint32_t NotSet = 0xffffffff;
  uint32_t mLabel = NotSet; ///< MC label encoding MCevent origin and fraction of correct contributors

 public:
  static constexpr int nbitsEvID = 16; // number of bits reserved for MC event ID
  static constexpr int nbitsSrcID = 7; // number of bits reserved for MC source ID
  static constexpr int nbitsCorrW = sizeof(mLabel) * 8 - nbitsEvID - nbitsSrcID - 1;

  // Mask to extract MC event ID
  static constexpr uint32_t MaskEvID = (0x1 << nbitsEvID) - 1;
  // Mask to extract MC source ID
  static constexpr uint32_t MaskSrcID = (0x1 << nbitsSrcID) - 1;
  // Mask to extract MC source and event ID only
  static constexpr uint32_t MaskSrcEvID = MaskSrcID | MaskEvID;
  // Mask to extract MC correct contribution weight
  static constexpr uint32_t MaskCorrW = (0x1 << nbitsCorrW) - 1;
  static constexpr float WeightNorm = 1. / float(MaskCorrW);

  MCEventLabel(int evID, int srcID, float corrw = 1.0) { set(evID, srcID, corrw); }
  MCEventLabel() = default;
  ~MCEventLabel() = default;

  // check if label was assigned
  bool isSet() const { return mLabel != NotSet; }
  // check if label was not assigned
  bool isEmpty() const { return mLabel == NotSet; }

  // conversion operator
  operator uint32_t() const { return mLabel; }
  // allow to retrieve bare label
  uint32_t getRawValue() const { return mLabel; }

  // get only combined identifier, discarding weight info
  uint32_t getIDOnly() const { return mLabel & MaskSrcEvID; }

  // compare
  bool compare(const MCEventLabel& other, bool strict = false) const
  {
    return strict ? (getRawValue() == other.getRawValue()) : (getIDOnly() == other.getIDOnly());
  }

  // comparison operator, compares only label, not eventual weight or correctness info
  bool operator==(const MCEventLabel& other) const { return compare(other); }

  // invalidate
  void unset() { mLabel = NotSet; }

  /// compose label
  void set(int evID, int srcID, float corrW)
  {
    uint32_t iw = static_cast<uint32_t>(std::round(corrW * MaskCorrW));
    assert(iw <= MaskCorrW);
    mLabel = (iw << (nbitsEvID + nbitsSrcID)) | ((MaskSrcID & static_cast<uint32_t>(srcID)) << nbitsEvID) | (MaskEvID & static_cast<uint32_t>(evID));
  }
  void setCorrWeight(float corrW)
  {
    uint32_t iw = static_cast<uint32_t>(std::round(corrW * MaskCorrW));
    assert(iw <= MaskCorrW);
    mLabel = (mLabel & ((MaskSrcID << nbitsEvID) | MaskEvID)) | (iw << (nbitsEvID + nbitsSrcID));
  }

  int getEventID() const { return mLabel & MaskEvID; }
  int getSourceID() const { return (mLabel >> nbitsEvID) & MaskSrcID; }
  float getCorrWeight() const { return ((mLabel >> (nbitsEvID + nbitsSrcID)) & MaskCorrW) * WeightNorm; }

  void get(int& evID, int& srcID, float& corrW)
  {
    /// parse label
    evID = getEventID();
    srcID = getSourceID();
    corrW = getCorrWeight();
  }

  void print() const;

  static constexpr uint32_t MaxSourceID() { return MaskSrcID; }
  static constexpr uint32_t MaxEventID() { return MaskEvID; }
  static constexpr float WeightPrecision() { return WeightNorm; }
  ClassDefNV(MCEventLabel, 1);
};
} // namespace o2

std::ostream& operator<<(std::ostream& os, const o2::MCEventLabel& c);

namespace std
{
// defining std::hash for MCEventLabel in order to be used with unordered_maps
template <>
struct hash<o2::MCEventLabel> {
 public:
  size_t operator()(o2::MCEventLabel const& label) const
  {
    return static_cast<uint32_t>(label);
  }
};
} // namespace std

#endif
