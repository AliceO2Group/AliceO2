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

#ifndef O2_MCH_MAPPING_CATHODESEGMENTATION_H
#define O2_MCH_MAPPING_CATHODESEGMENTATION_H

#include "CathodeSegmentationCInterface.h"
#include <stdexcept>
#include <memory>
#include <iostream>
#include <vector>
#include <boost/format.hpp>

namespace o2
{
namespace mch
{
namespace mapping
{

/// @brief A CathodeSegmentation lets you _find_ pads on a given plane (cathode) of a detection element
/// and then _inspect_ those pads.
///
/// Pads can be found by :
/// - their position (x,y) in the plane
/// - their front-end electronics characteristics (which DualSampa chip, which chip channel)
///
/// What you get back from the find methods are not pad objects directly but a reference (integer)
/// that can then be used to query the segmentation object about that pad.
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

class CathodeSegmentation
{
 public:
  /// This ctor throws if detElemId is invalid
  CathodeSegmentation(int detElemId, bool isBendingPlane)
    : mImpl{ mchCathodeSegmentationConstruct(detElemId, isBendingPlane) },
      mDualSampaIds{},
      mDetElemId{ detElemId },
      mIsBendingPlane{ isBendingPlane }
  {
    if (!mImpl) {
      throw std::runtime_error("Can not create segmentation for DE " + std::to_string(detElemId) +
                               (mIsBendingPlane ? " Bending" : " NonBending"));
    }
    std::vector<int> dpid;
    auto addDualSampaId = [&dpid](int dualSampaId) { dpid.push_back(dualSampaId); };
    auto callback = [](void* data, int dualSampaId) {
      auto fn = static_cast<decltype(&addDualSampaId)>(data);
      (*fn)(dualSampaId);
    };
    mchCathodeSegmentationForEachDualSampa(mImpl, callback, &addDualSampaId);
    mDualSampaIds = dpid;
  }

  bool operator==(const CathodeSegmentation& rhs) const
  {
    return mDetElemId == rhs.mDetElemId && mIsBendingPlane == rhs.mIsBendingPlane;
  }

  bool operator!=(const CathodeSegmentation& rhs) const { return !(rhs == *this); }

  friend void swap(CathodeSegmentation& a, CathodeSegmentation& b)
  {
    using std::swap;

    swap(a.mImpl, b.mImpl);
    swap(a.mDualSampaIds, b.mDualSampaIds);
    swap(a.mDetElemId, b.mDetElemId);
    swap(a.mIsBendingPlane, b.mIsBendingPlane);
  }

  CathodeSegmentation(const CathodeSegmentation& seg)
  {
    mDetElemId = seg.mDetElemId;
    mIsBendingPlane = seg.mIsBendingPlane;
    mImpl = mchCathodeSegmentationConstruct(mDetElemId, mIsBendingPlane);
    mDualSampaIds = seg.mDualSampaIds;
  }

  CathodeSegmentation& operator=(CathodeSegmentation seg)
  {
    swap(*this, seg);
    return *this;
  }

  ~CathodeSegmentation() { mchCathodeSegmentationDestruct(mImpl); }

  /** @name Pad Unique Identifier
   * Pads are identified by a unique integer, paduid.
   * @warning This paduid is only valid within the realm of this CathodeSegmentation object
   * (to use the query methods padPosition, padSize, etc...).
   * So do _not_ rely on any given value it might take (as it might change
   * between e.g. library version or underlying implementation)
   */
  ///@{ Not every integer is a valid paduid. This method will tell if paduid is a valid one.
  bool isValid(int paduid) const { return mchCathodeSegmentationIsPadValid(mImpl, paduid) > 0; }
  ///@}

  /** @name Pad finding.
   * Methods to find a pad.
   * In each case the returned integer
   * represents either a paduid if a pad is found or
   * an integer representing an invalid paduid otherwise.
   * Validity of the returned value can be tested using isValid()
   */
  ///@{
  /** Find the pad at position (x,y) (in cm). */
  int findPadByPosition(double x, double y) const { return mchCathodeSegmentationFindPadByPosition(mImpl, x, y); }

  /** Find the pad connected to the given channel of the given dual sampa. */
  int findPadByFEE(int dualSampaId, int dualSampaChannel) const
  {
    if (dualSampaChannel < 0 || dualSampaChannel > 63) {
      throw std::out_of_range("dualSampaChannel should be between 0 and 63");
    }
    return mchCathodeSegmentationFindPadByFEE(mImpl, dualSampaId, dualSampaChannel);
  }
  ///@}

  /// @name Pad information retrieval.
  /// Given a _valid_ paduid those methods return information
  /// (position, size, fee) about that pad.
  /// @{
  double padPositionX(int paduid) const { return mchCathodeSegmentationPadPositionX(mImpl, paduid); }

  double padPositionY(int paduid) const { return mchCathodeSegmentationPadPositionY(mImpl, paduid); }

  double padSizeX(int paduid) const { return mchCathodeSegmentationPadSizeX(mImpl, paduid); }

  double padSizeY(int paduid) const { return mchCathodeSegmentationPadSizeY(mImpl, paduid); }

  int padDualSampaId(int paduid) const { return mchCathodeSegmentationPadDualSampaId(mImpl, paduid); }

  int padDualSampaChannel(int paduid) const { return mchCathodeSegmentationPadDualSampaChannel(mImpl, paduid); }

  /// @}

  /** @name Some general characteristics of this segmentation. */
  ///@{
  bool isBendingPlane() const { return mIsBendingPlane; }

  int nofDualSampas() const { return mDualSampaIds.size(); }

  int nofPads() const
  {
    int n{ 0 };
    for (auto i = 0; i < nofDualSampas(); ++i) {
      forEachPadInDualSampa(dualSampaId(i), [&n](int /*paduid*/) { ++n; });
    }
    return n;
  }

  /// \param dualSampaIndex must be in the range 0..nofDualSampas()-1
  /// \return the DualSampa chip id for a given index
  int dualSampaId(int dualSampaIndex) const { return mDualSampaIds[dualSampaIndex]; }
  ///@}

  /** @name ForEach methods.
   * Those methods let you execute a function on each of the pads belonging to
   * some group.
   */
  ///@{
  template <typename CALLABLE>
  void forEachPadInDualSampa(int dualSampaId, CALLABLE&& func) const;

  template <typename CALLABLE>
  void forEachPadInArea(double xmin, double ymin, double xmax, double ymax, CALLABLE&& func) const;

  template <typename CALLABLE>
  void forEachNeighbouringPad(int paduid, CALLABLE&& func) const;
  ///@}
 private:
  MchCathodeSegmentationHandle mImpl;
  std::vector<int> mDualSampaIds;
  int mDetElemId;
  bool mIsBendingPlane;
};

template <typename CALLABLE>
void CathodeSegmentation::forEachPadInDualSampa(int dualSampaId, CALLABLE&& func) const
{
  auto callback = [](void* data, int paduid) {
    auto fn = static_cast<decltype(&func)>(data);
    (*fn)(paduid);
  };
  mchCathodeSegmentationForEachPadInDualSampa(mImpl, dualSampaId, callback, &func);
}

template <typename CALLABLE>
void CathodeSegmentation::forEachPadInArea(double xmin, double ymin, double xmax, double ymax, CALLABLE&& func) const
{
  auto callback = [](void* data, int paduid) {
    auto fn = static_cast<decltype(&func)>(data);
    (*fn)(paduid);
  };
  mchCathodeSegmentationForEachPadInArea(mImpl, xmin, ymin, xmax, ymax, callback, &func);
}

template <typename CALLABLE>
void CathodeSegmentation::forEachNeighbouringPad(int paduid, CALLABLE&& func) const
{
  auto callback = [](void* data, int puid) {
    auto fn = static_cast<decltype(&func)>(data);
    (*fn)(puid);
  };
  mchCathodeSegmentationForEachNeighbouringPad(mImpl, paduid, callback, &func);
}

/** Convenience method to loop over detection elements. */
template <typename CALLABLE>
void forEachDetectionElement(CALLABLE&& func)
{
  auto callback = [](void* data, int detElemId) {
    auto fn = static_cast<decltype(&func)>(data);
    (*fn)(detElemId);
  };
  mchCathodeSegmentationForEachDetectionElement(callback, &func);
}

/** Convenience method to loop over all segmentation types. */
template <typename CALLABLE>
void forOneDetectionElementOfEachCathodeSegmentationType(CALLABLE&& func)
{
  auto callback = [](void* data, int detElemId) {
    auto fn = static_cast<decltype(&func)>(data);
    (*fn)(detElemId);
  };
  mchCathodeSegmentationForOneDetectionElementOfEachCathodeSegmentationType(callback, &func);
}

/** Convenience method to get a string representation of a pad. */
inline std::string padAsString(const CathodeSegmentation& seg, int paduid)
{
  if (seg.isValid(paduid)) {
    return boost::str(boost::format("Pad %10d FEC %4d CH %2d X %7.3f Y %7.3f SX %7.3f SY %7.3f") % paduid %
                      seg.padDualSampaId(paduid) % seg.padDualSampaChannel(paduid) % seg.padPositionX(paduid) %
                      seg.padPositionY(paduid) % seg.padSizeX(paduid) % seg.padSizeY(paduid));
  } else {
    return "invalid pad with uid=" + std::to_string(paduid);
  }
}

} // namespace mapping
} // namespace mch
} // namespace o2
#endif
