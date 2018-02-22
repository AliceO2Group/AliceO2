//
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


#ifndef O2_MCH_MAPPING_IMPL3_SEGMENTATION_H
#define O2_MCH_MAPPING_IMPL3_SEGMENTATION_H

#include "PadGroup.h"
#include "PadGroupType.h"
#include <vector>
#include <set>
#include <ostream>
#include <boost/geometry/index/rtree.hpp>

namespace o2 {
namespace mch {
namespace mapping {
namespace impl3 {

class Segmentation
{
  public:

    static constexpr int InvalidPadUid{-1};

    using Point = boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian>;
    using Box = boost::geometry::model::box<Point>;
    using Value = std::pair<Box, unsigned>;

    Segmentation(int segType, bool isBendingPlane, std::vector<PadGroup> padGroups);

    Segmentation(int segType, bool isBendingPlane, std::vector<PadGroup> padGroups,
                 std::vector<PadGroupType> padGroupTypes,
                 std::vector<std::pair<float, float>> padSizes);

    /// Return the list of paduids for the pads of the given dual sampa.
    std::vector<int> getPadUids(int dualSampaIds) const;

    /// Return the list of paduids for the pads contained in the box {xmin,ymin,xmax,ymax}.
    std::vector<int> getPadUids(double xmin, double ymin, double xmax, double ymax) const;

    /// Return the list of paduids of the pads which are neighbours to paduid
    std::vector<int> getNeighbouringPadUids(int paduid) const;

    std::set<int> dualSampaIds() const
    { return mDualSampaIds; }

    int findPadByPosition(double x, double y) const;

    int findPadByFEE(int dualSampaId, int dualSampaChannel) const;

    bool hasPadByPosition(double x, double y) const
    { return findPadByPosition(x, y) != InvalidPadUid; }

    bool hasPadByFEE(int dualSampaId, int dualSampaChannel) const
    { return findPadByFEE(dualSampaId, dualSampaChannel) != InvalidPadUid; }

    friend std::ostream &operator<<(std::ostream &os, const Segmentation &seg);

    double padPositionX(int paduid) const;

    double padPositionY(int paduid) const;

    double padSizeX(int paduid) const;

    double padSizeY(int paduid) const;

    int padDualSampaId(int paduid) const;

    int padDualSampaChannel(int paduid) const;

  private:
    int dualSampaIndex(int dualSampaId) const;

    void fillRtree();

    std::ostream &showPad(std::ostream &out, int index) const;

    const PadGroup &padGroup(int paduid) const;

    const PadGroupType &padGroupType(int paduid) const;

    double squaredDistance(int paduid, double x, double y) const;

  private:
    int mSegType;
    bool mIsBendingPlane;
    std::vector<PadGroup> mPadGroups;
    std::set<int> mDualSampaIds;
    std::vector<PadGroupType> mPadGroupTypes;
    std::vector<std::pair<float, float>> mPadSizes;
    boost::geometry::index::rtree<Value, boost::geometry::index::quadratic<8>> mRtree;
    std::vector<int> mPadUid2PadGroupIndex;
    std::vector<int> mPadUid2PadGroupTypeFastIndex;
    std::vector<int> mPadGroupIndex2PadUidIndex;

};

Segmentation *createSegmentation(int detElemId, bool isBendingPlane);
}
}
}
}
#endif
