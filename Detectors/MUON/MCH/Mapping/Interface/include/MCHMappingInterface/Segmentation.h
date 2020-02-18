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
  Segmentation(int deid);

  /** @name Some general characteristics of this segmentation. */
  ///@{
  int detElemId() const;
  int nofPads() const;
  int nofDualSampas() const;
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

  /// Loop over dual sampas of this detection element
  void forEachDualSampa(std::function<void(int dualSampaId)> func) const;

  /** @name ForEach (pad) methods.
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

  /** @name Access to individual cathode segmentations. 
    * Not needed in most cases.
    */
  ///@{
  const CathodeSegmentation& bending() const;
  const CathodeSegmentation& nonBending() const;
  ///@}

  bool operator==(const Segmentation& rhs) const;
  bool operator!=(const Segmentation& rhs) const;
  friend void swap(Segmentation& a, Segmentation& b);
  Segmentation(const Segmentation& seg);
  Segmentation(const Segmentation&& seg);
  Segmentation& operator=(Segmentation seg);

 private:
  int padC2DE(int catPadIndex, bool isBending) const;
  void catSegPad(int dePadIndex, const CathodeSegmentation*& catseg, int& padcuid) const;

 private:
  int mDetElemId;
  CathodeSegmentation mBending;
  CathodeSegmentation mNonBending;
  int mPadIndexOffset = 0;
};

/// segmentation(int) is a convenience function that
/// returns a segmentation for a given detection element.
///
/// That segmentation is part of a pool of Segmentations
/// (one per DE) handled by this module.
///
/// Note that this may be not what you want as this module
/// will always create one (but only one) Segmentation for
/// all 156 detection elements at once, even when only one
/// Segmentation is requested.
///
/// For instance, if you know you'll be dealing with only
/// one detection element, you'd be better using
/// the Segmentation ctor simply, and ensure by yourself
/// that you are only creating it once in order not to incur
/// the (high) price of the construction time of that Segmentation.
const Segmentation& segmentation(int detElemId);

} // namespace mapping
} // namespace mch
} // namespace o2

#include "Segmentation.inl"

#endif
