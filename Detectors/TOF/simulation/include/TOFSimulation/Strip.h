// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  TOF strip class: it will be used to store the hits and digits at TOF that
//  fall in the same strip
//

#ifndef ALICEO2_TOF_STRIP_H_
#define ALICEO2_TOF_STRIP_H_

#include <TOFBase/Digit.h>
#include <TObject.h>
#include <exception>
#include <map>
#include <sstream>
#include <vector>
#include "MathUtils/Cartesian3D.h"
#include "TOFSimulation/Detector.h" // for HitType

namespace o2
{
namespace tof
{

/// @class Strip
/// @brief Container for similated points connected to a given TOF strip
/// This will be used in order to allow a more efficient clusterization
/// that can happen only between digits that belong to the same strip
///

class Strip
{

 public:
  /// Default constructor
  Strip() = default;

  /// Destructor
  ~Strip() = default;

  /// Main constructor
  /// @param stripindex Index of the strip
  /// @param mat Transformation matrix
  Strip(Int_t index);

  /// Copy constructor
  /// @param ref Reference for the copy
  Strip(const Strip& ref) = default;

  /// Empties the point container
  /// @param option unused
  void clear();

  /// Change the chip index
  /// @param index New chip index
  void setStripIndex(Int_t index) { mStripIndex = index; }
  void init(Int_t index)
  //, const o2::Transform3D* mat)
  {
    mStripIndex = index;
    //mMat = mat;
  }

  /// Get the chip index
  /// @return Index of the chip
  Int_t getStripIndex() const { return mStripIndex; }
  /// Insert new ITSMFT point into the Chip
  /// @param p Hit to be added
  void insertHit(const o2::tof::HitType* p);

  /// Get the number of point assigned to the chip
  /// @return Number of points assigned to the chip
  Int_t getNumberOfHits() const { return mHits.size(); }
  Int_t getNumberOfDigits() const { return mDigits.size(); }

  /// Get the strip index from hit
  Int_t getStripIndex(const o2::tof::HitType* hit);

  /// reset points container
  void clearHits() { mHits.clear(); }
  o2::tof::Digit* findDigit(ULong64_t key);

  /// Access Hit assigned to chip at a given index
  /// @param index Index of the point
  /// @return Hit at given index (nullptr if index is out of bounds)
  inline const o2::tof::HitType* getHitAt(Int_t index) const { return mHits.at(index); }

  Int_t addDigit(Int_t channel, Int_t tdc, Int_t tot, Int_t bc, Int_t lbl); // returns the MC label

  void fillOutputContainer(std::vector<o2::tof::Digit>& digits);

 protected:
  Int_t mStripIndex = -1;                      ///< Strip ID
  std::vector<const o2::tof::HitType*> mHits;  ///< Hits connected to the given strip
  std::map<ULong64_t, o2::tof::Digit> mDigits; ///< Map of fired digits, possibly in multiple frames

  ClassDefNV(Strip, 1);
};

inline o2::tof::Digit* Strip::findDigit(ULong64_t key)
{
  // finds the digit corresponding to global key
  auto digitentry = mDigits.find(key);
  return digitentry != mDigits.end() ? &(digitentry->second) : nullptr;
}

} // namespace tof
} // namespace o2

#endif /* defined(ALICEO2_TOF_STRIP_H_) */
