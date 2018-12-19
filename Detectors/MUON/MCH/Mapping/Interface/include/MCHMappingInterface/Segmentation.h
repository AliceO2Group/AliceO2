// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file CathodeSegmentation.h
 * C++ Interface to the Muon MCH mapping.
 * @author  Laurent Aphecetche
 */

#ifndef O2_MCH_MAPPING_SEGMENTATION_H
#define O2_MCH_MAPPING_SEGMENTATION_H

#include "MCHMappingInterface/CathodeSegmentation.h"

namespace o2
{
namespace mch
{
namespace mapping
{

/// @brief A Segmentation lets you _find_ pads of a detection element
/// and then _inspect_ those pads.
///
/// Note that this class is closely related to the CathodeSegmentation one which
/// only deals with one of the two cathodes of a detection element.
///
/// Pads can be found by :
/// - their position (x,y)
/// - their front-end electronics characteristics (which DualSampa chip, which chip channel)
///
/// What you get back from the find methods are not pad objects directly but a reference (integer)
/// that can then be used to query the segmentation object about that pad.
/// @warning This integer reference (dePadIndex) is only valid within the realm
/// of this Segmentation object (to use the query methods padPosition, padSize, etc...).
/// So do _not_ rely on any given value it might take (as it might change
/// between e.g. library version or underlying implementation)
///
/// By convention, the pad references are contiguous (ranging from 0 to number
/// of pads in the detection element), and the bending pads come first.
///
/// Pad information that can be retrieved :
/// - position in x and y directions
/// - size in x and y directions
/// - dual sampa id
/// - dual sampa channel
///
/// In addition, you can _apply_ some function to a group of pads using one of the the forEach methods :
/// - all the pads belonging to a given dual sampa
/// - all the pads within a given area (box)
/// - all the pads that are neighbours of a given pad

class Segmentation
{
 public:
  /// This ctor throws if deid is invalid
  Segmentation(int deid) : mDetElemId{ deid }, mBending{ CathodeSegmentation(deid, true) }, mNonBending{ CathodeSegmentation(deid, false) }, mPadIndexOffset{ mBending.nofPads() } {}

  bool operator==(const Segmentation& rhs) const
  {
    return mDetElemId == rhs.mDetElemId;
  }

  bool operator!=(const Segmentation& rhs) const { return !(rhs == *this); }

  friend void swap(Segmentation& a, Segmentation& b)
  {
    using std::swap;
    swap(a.mDetElemId, b.mDetElemId);
  }

  Segmentation(const Segmentation& seg) : mBending{ seg.mBending }, mNonBending{ seg.mNonBending }
  {
    mDetElemId = seg.mDetElemId;
  }

  Segmentation(const Segmentation&& seg) : mBending{ std::move(seg.mBending) }, mNonBending{ std::move(seg.mNonBending) }
  {
    mDetElemId = seg.mDetElemId;
  }
  Segmentation& operator=(Segmentation seg)
  {
    swap(*this, seg);
    return *this;
  }

  /** @name Some general characteristics of this segmentation. */
  ///@{
  int detElemId() const { return mDetElemId; }
  int nofPads() const { return mBending.nofPads() + mNonBending.nofPads(); }
  int nofDualSampas() const { return mBending.nofDualSampas() + mNonBending.nofDualSampas(); }
  ///@}

  /** @name Access to individual cathode segmentations. 
    * Not needed in most cases.
    */
  ///@{
  const CathodeSegmentation& Bending() const { return mBending; }
  const CathodeSegmentation& NonBending() const { return mNonBending; }
  ///@}

  /** @name Pad finding.
   * Methods to find a pad.
   * In each case the returned integer(s)
   * represents either a dePadIndex if a pad is found or
   * an integer representing an invalid dePadIndex otherwise.
   * Validity of the returned value can be tested using isValid()
   */
  ///@{
  /** Find the pads at position (x,y) (in cm). 
    Returns true is the bpad and nbpad has been filled with a valid dePadIndex,
    false otherwise (if position is outside the segmentation area).
    @param bpad the dePadIndex of the bending pad at position (x,y)
    @param nbpad the dePadIndex of the non-bending pad at position (x,y)
  */
  bool findPadPairByPosition(double x, double y, int& bpad, int& nbpad) const;

  /** Find the pad connected to the given channel of the given dual sampa. */
  int findPadByFEE(int dualSampaId, int dualSampaChannel) const;
  ///@}

  /// @name Pad information retrieval.
  /// Given a _valid_ dePadIndex those methods return information
  /// (position, size, fee, etc...) about that pad.
  /// @{
  double padPositionX(int dePadIndex) const;
  double padPositionY(int dePadIndex) const;
  double padSizeX(int dePadIndex) const;
  double padSizeY(int dePadIndex) const;
  int padDualSampaId(int dePadIndex) const;
  int padDualSampaChannel(int dePadIndex) const;
  bool isValid(int dePadIndex) const;
  bool isBendingPad(int dePadIndex) const { return dePadIndex < mPadIndexOffset; }
  std::string padAsString(int dePadIndex) const;
  /// @}

  /** @name ForEach methods.
   * Those methods let you execute a function on each of the pads belonging to
   * some group.
   */
  ///@{
  template <typename CALLABLE>
  void forEachPad(CALLABLE&& func) const;

  template <typename CALLABLE>
  void forEachNeighbouringPad(int dePadIndex, CALLABLE&& func) const;

  template <typename CALLABLE>
  void forEachPadInArea(double xmin, double ymin, double xmax, double ymax, CALLABLE&& func) const;
  ///@}

 private:
  int padC2DE(int catPadIndex, bool isBending) const;
  void catSegPad(int dePadIndex, const CathodeSegmentation*& catseg, int& padcuid) const;

 private:
  int mDetElemId;
  CathodeSegmentation mBending;
  CathodeSegmentation mNonBending;
  int mPadIndexOffset = 0;
};

inline int Segmentation::findPadByFEE(int dualSampaId, int dualSampaChannel) const
{
  bool isBending = dualSampaId < 1024;
  int catPadIndex;
  if (isBending) {
    catPadIndex = mBending.findPadByFEE(dualSampaId, dualSampaChannel);
    if (!mBending.isValid(catPadIndex)) {
      return -1;
    }
  } else {
    catPadIndex = mNonBending.findPadByFEE(dualSampaId, dualSampaChannel);
    if (!mNonBending.isValid(catPadIndex)) {
      return -1;
    }
  }
  return padC2DE(catPadIndex, isBending);
}

inline bool Segmentation::isValid(int dePadIndex) const
{
  if (dePadIndex < mPadIndexOffset) {
    return mBending.isValid(dePadIndex);
  }
  return mNonBending.isValid(dePadIndex - mPadIndexOffset);
}

inline int Segmentation::padC2DE(int catPadIndex, bool isBending) const
{
  if (isBending) {
    return catPadIndex;
  }
  return catPadIndex + mPadIndexOffset;
}

inline void Segmentation::catSegPad(int dePadIndex, const CathodeSegmentation*& catseg, int& padcuid) const
{
  if (!isValid(dePadIndex)) {
    catseg = nullptr;
    return;
  }
  if (dePadIndex < mPadIndexOffset) {
    catseg = &mBending;
    padcuid = dePadIndex;
    return;
  }
  catseg = &mNonBending;
  padcuid = dePadIndex - mPadIndexOffset;
}

inline bool Segmentation::findPadPairByPosition(double x, double y, int& b, int& nb) const
{
  b = mBending.findPadByPosition(x, y);
  nb = mNonBending.findPadByPosition(x, y);
  if (!mBending.isValid(b) || !mNonBending.isValid(nb)) {
    return false;
  }
  b = padC2DE(b, true);
  nb = padC2DE(nb, false);
  return true;
}

template <typename CALLABLE>
void Segmentation::forEachPad(CALLABLE&& func) const
{
  mBending.forEachPad(func);
  int offset{ mPadIndexOffset };
  mNonBending.forEachPad([&offset, &func](int catPadIndex) {
    func(catPadIndex + offset);
  });
}

template <typename CALLABLE>
void Segmentation::forEachPadInArea(double xmin, double ymin, double xmax, double ymax, CALLABLE&& func) const
{
  mBending.forEachPadInArea(xmin, ymin, xmax, ymax, func);
  int offset{ mPadIndexOffset };
  mNonBending.forEachPadInArea(xmin, ymin, xmax, ymax, [&offset, &func](int catPadIndex) {
    func(catPadIndex + offset);
  });
}

template <typename CALLABLE>
void Segmentation::forEachNeighbouringPad(int dePadIndex, CALLABLE&& func) const
{
  const CathodeSegmentation* catSeg{ nullptr };
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);

  int offset{ 0 };
  if (!isBendingPad(dePadIndex)) {
    offset = mPadIndexOffset;
  }
  catSeg->forEachNeighbouringPad(catPadIndex, [&offset, &func](int cindex) {
    func(cindex + offset);
  });
}

inline std::string Segmentation::padAsString(int dePadIndex) const
{
  if (!isValid(dePadIndex)) {
    return "invalid pad with index=" + std::to_string(dePadIndex);
  }
  const CathodeSegmentation* catSeg{ nullptr };
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padAsString(catPadIndex);
}

inline int Segmentation::padDualSampaId(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{ nullptr };
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padDualSampaId(catPadIndex);
}

inline int Segmentation::padDualSampaChannel(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{ nullptr };
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padDualSampaChannel(catPadIndex);
}

inline double Segmentation::padPositionX(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{ nullptr };
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padPositionX(catPadIndex);
}

inline double Segmentation::padPositionY(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{ nullptr };
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padPositionY(catPadIndex);
}

inline double Segmentation::padSizeX(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{ nullptr };
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padSizeX(catPadIndex);
}

inline double Segmentation::padSizeY(int dePadIndex) const
{
  const CathodeSegmentation* catSeg{ nullptr };
  int catPadIndex;
  catSegPad(dePadIndex, catSeg, catPadIndex);
  return catSeg->padSizeY(catPadIndex);
}

} // namespace mapping
} // namespace mch
} // namespace o2

#endif
