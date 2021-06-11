// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#ifndef O2_MCH_MAPPING_Impl4_CATHODESEGMENTATION_H
#define O2_MCH_MAPPING_Impl4_CATHODESEGMENTATION_H

#include "PadGroup.h"
#include "PadGroupType.h"
#include <vector>
#include <set>
#include <ostream>
#include <boost/geometry/index/rtree.hpp>

namespace o2
{
namespace mch
{
namespace mapping
{
namespace impl4
{

class CathodeSegmentation
{
 public:
  static constexpr int InvalidCatPadIndex{-1};

  using Point =
    boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian>;
  using Box = boost::geometry::model::box<Point>;
  using Value = std::pair<Box, unsigned>;

  CathodeSegmentation(int segType, bool isBendingPlane,
                      std::vector<PadGroup> padGroups,
                      std::vector<PadGroupType> padGroupTypes,
                      std::vector<std::pair<float, float>> padSizes);

  /// Return the list of catPadIndexs for the pads of the given dual sampa.
  std::vector<int> getCatPadIndexs(int dualSampaIds) const;

  /// Return the list of catPadIndexs for the pads contained in the box
  /// {xmin,ymin,xmax,ymax}.
  std::vector<int> getCatPadIndexs(double xmin, double ymin, double xmax,
                                   double ymax) const;

  /// Return the list of catPadIndexs of the pads which are neighbours to
  /// catPadIndex
  std::vector<int> getNeighbouringCatPadIndexs(int catPadIndex) const;

  std::set<int> dualSampaIds() const { return mDualSampaIds; }

  int findPadByPosition(double x, double y) const;

  int findPadByFEE(int dualSampaId, int dualSampaChannel) const;

  bool hasPadByPosition(double x, double y) const
  {
    return findPadByPosition(x, y) != InvalidCatPadIndex;
  }

  bool hasPadByFEE(int dualSampaId, int dualSampaChannel) const
  {
    return findPadByFEE(dualSampaId, dualSampaChannel) != InvalidCatPadIndex;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const CathodeSegmentation& seg);

  double padPositionX(int catPadIndex) const;

  double padPositionY(int catPadIndex) const;

  double padSizeX(int catPadIndex) const;

  double padSizeY(int catPadIndex) const;

  int padDualSampaId(int catPadIndex) const;

  int padDualSampaChannel(int catPadIndex) const;

  bool isValid(int catPadIndex) const;

 private:
  int dualSampaIndex(int dualSampaId) const;

  void fillRtree();

  std::ostream& showPad(std::ostream& out, int index) const;

  const PadGroup& padGroup(int catPadIndex) const;

  const PadGroupType& padGroupType(int catPadIndex) const;

  double squaredDistance(int catPadIndex, double x, double y) const;

 private:
  int mSegType;
  bool mIsBendingPlane;
  std::vector<PadGroup> mPadGroups;
  std::set<int> mDualSampaIds;
  std::vector<PadGroupType> mPadGroupTypes;
  std::vector<std::pair<float, float>> mPadSizes;
  boost::geometry::index::rtree<Value, boost::geometry::index::quadratic<8>>
    mRtree;
  std::vector<int> mCatPadIndex2PadGroupIndex;
  std::vector<int> mCatPadIndex2PadGroupTypeFastIndex;
  std::vector<int> mPadGroupIndex2CatPadIndexIndex;
};

CathodeSegmentation* createCathodeSegmentation(int detElemId,
                                               bool isBendingPlane);
} // namespace impl4
} // namespace mapping
} // namespace mch
} // namespace o2
#endif
