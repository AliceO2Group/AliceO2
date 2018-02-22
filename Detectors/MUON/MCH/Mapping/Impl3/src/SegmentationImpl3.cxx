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

#include "SegmentationImpl3.h"
#include "boost/format.hpp"
#include "GenDetElemId2SegType.h"
#include "PadGroup.h"
#include "PadSize.h"
#include "MCHMappingInterface/Segmentation.h"
#include "SegmentationCreator.h"
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <gsl/gsl>

using namespace o2::mch::mapping::impl3;

namespace o2 {
namespace mch {
namespace mapping {
namespace impl3 {

Segmentation *createSegmentation(int detElemId, bool isBendingPlane)
{
  int segType = detElemId2SegType(detElemId);
  SegmentationCreator creator = getSegmentationCreator(segType);
  if (creator == nullptr) {
    return nullptr;
  }
  return creator(isBendingPlane);
}

void Segmentation::fillRtree()
{
  int paduid{0};

  for (auto padGroupIndex = 0; padGroupIndex < mPadGroups.size(); ++padGroupIndex) {
    mPadGroupIndex2PadUidIndex.push_back(paduid);
    auto &pg = mPadGroups[padGroupIndex];
    auto &pgt = mPadGroupTypes[pg.mPadGroupTypeId];
    double dx{mPadSizes[pg.mPadSizeId].first};
    double dy{mPadSizes[pg.mPadSizeId].second};
    for (int ix = 0; ix < pgt.getNofPadsX(); ++ix) {
      for (int iy = 0; iy < pgt.getNofPadsY(); ++iy) {
        if (pgt.id(ix, iy) >= 0) {

          double xmin = ix * dx + pg.mX;
          double xmax = (ix + 1) * dx + pg.mX;
          double ymin = iy * dy + pg.mY;
          double ymax = (iy + 1) * dy + pg.mY;

          mRtree.insert(std::make_pair(Segmentation::Box{
            Segmentation::Point(xmin, ymin),
            Segmentation::Point(xmax, ymax)
          }, paduid));

          mPadUid2PadGroupIndex.push_back(padGroupIndex);
          mPadUid2PadGroupTypeFastIndex.push_back(pgt.fastIndex(ix, iy));
          ++paduid;
        }
      }
    }
  }
}

std::set<int> getUnique(const std::vector<PadGroup> &padGroups)
{
  // extract from padGroup vector the unique integer values given by func
  std::set<int> u;
  for (auto &pg: padGroups) {
    u.insert(pg.mFECId);
  }
  return u;
}

#if 0
void dump(const std::string &msg, const std::vector<int> &v)
{
  std::cout << msg << " of size " << v.size() << " : ";

  for (auto &value: v) {
    std::cout << boost::format("%3d") % value << ",";
  }
  std::cout << "\n";
}
#endif

Segmentation::Segmentation(int segType, bool isBendingPlane, std::vector<PadGroup> padGroups,
                           std::vector<PadGroupType> padGroupTypes,
                           std::vector<std::pair<float, float>> padSizes)
  :
  mSegType{segType},
  mIsBendingPlane{isBendingPlane},
  mPadGroups{std::move(padGroups)},
  mDualSampaIds{getUnique(mPadGroups)},
  mPadGroupTypes{std::move(padGroupTypes)},
  mPadSizes{std::move(padSizes)},
  mPadUid2PadGroupIndex{},
  mPadUid2PadGroupTypeFastIndex{},
  mPadGroupIndex2PadUidIndex{}
{
  fillRtree();
}

std::vector<int> Segmentation::getPadUids(int dualSampaId) const
{
  std::vector<int> pi;

  for (auto padGroupIndex = 0; padGroupIndex < mPadGroups.size(); ++padGroupIndex) {
    if (mPadGroups[padGroupIndex].mFECId == dualSampaId) {
      auto &pgt = mPadGroupTypes[mPadGroups[padGroupIndex].mPadGroupTypeId];
      auto i1 = mPadGroupIndex2PadUidIndex[padGroupIndex];
      for (auto i = i1; i < i1 + pgt.getNofPads(); ++i) {
        pi.push_back(i);
      }
    }
  }

  return pi;
}

std::vector<int> Segmentation::getPadUids(double xmin, double ymin, double xmax, double ymax) const
{
  std::vector<Segmentation::Value> result_n;
  mRtree.query(boost::geometry::index::intersects(Segmentation::Box({xmin, ymin}, {xmax, ymax})),
               std::back_inserter(result_n));
  std::vector<int> paduids;
  for (auto &r: result_n) {
    paduids.push_back(r.second);
  }
  return paduids;
}

std::vector<int> Segmentation::getNeighbouringPadUids(int paduid) const
{
  double x = padPositionX(paduid);
  double y = padPositionY(paduid);
  double dx = padSizeX(paduid) / 2.0;
  double dy = padSizeY(paduid) / 2.0;

  const double offset{0.1}; // 1 mm

  auto pads = getPadUids(x - dx - offset, y - dy - offset, x + dx + offset, y + dy + offset);
  pads.erase(std::remove(begin(pads), end(pads), paduid), end(pads));
  return pads;
}

double Segmentation::squaredDistance(int paduid, double x, double y) const
{
  double px = padPositionX(paduid) - x;
  double py = padPositionY(paduid) - y;
  return px * px + py * py;
}

int Segmentation::findPadByPosition(double x, double y) const
{
  const double epsilon{1E-4};
  auto pads = getPadUids(x - epsilon, y - epsilon, x + epsilon, y + epsilon);

  double dmin{std::numeric_limits<double>::max()};
  int paduid{InvalidPadUid};

  for (auto i = 0; i < pads.size(); ++i) {
    double d{squaredDistance(pads[i], x, y)};
    if (d < dmin) {
      paduid = pads[i];
      dmin = d;
    }
  }

  return paduid;
}

const PadGroup &Segmentation::padGroup(int paduid) const
{
  return gsl::at(mPadGroups, mPadUid2PadGroupIndex[paduid]);
}

const PadGroupType &Segmentation::padGroupType(int paduid) const
{
  return gsl::at(mPadGroupTypes, padGroup(paduid).mPadGroupTypeId);
}

int Segmentation::findPadByFEE(int dualSampaId, int dualSampaChannel) const
{
  for (auto paduid: getPadUids(dualSampaId)) {
    if (padGroupType(paduid).id(mPadUid2PadGroupTypeFastIndex[paduid]) == dualSampaChannel) {
      return paduid;
    }
  }
  return InvalidPadUid;
}

double Segmentation::padPositionX(int paduid) const
{
  auto &pg = padGroup(paduid);
  auto &pgt = padGroupType(paduid);
  return pg.mX + (pgt.ix(mPadUid2PadGroupTypeFastIndex[paduid]) + 0.5) * mPadSizes[pg.mPadSizeId].first;
}

double Segmentation::padPositionY(int paduid) const
{
  auto &pg = padGroup(paduid);
  auto &pgt = padGroupType(paduid);
  return pg.mY + (pgt.iy(mPadUid2PadGroupTypeFastIndex[paduid]) + 0.5) * mPadSizes[pg.mPadSizeId].second;
}

double Segmentation::padSizeX(int paduid) const
{
  return mPadSizes[padGroup(paduid).mPadSizeId].first;
}

double Segmentation::padSizeY(int paduid) const
{
  return mPadSizes[padGroup(paduid).mPadSizeId].second;
}

int Segmentation::padDualSampaId(int paduid) const
{
  return padGroup(paduid).mFECId;
}

int Segmentation::padDualSampaChannel(int paduid) const
{
  return padGroupType(paduid).id(mPadUid2PadGroupTypeFastIndex[paduid]);
}

std::ostream &operator<<(std::ostream &out, const std::pair<float, float> &p)
{
  out << p.first << "," << p.second;
  return out;
}

template<typename T>
void dump(std::ostream &out, const std::string &msg, const std::vector<T> &v, int n)
{

  out << msg << "\n";
  for (auto i = 0; i < n; ++i) {
    if (i < v.size()) {
      out << v[i] << "\n";
    }
  }
}

std::ostream &operator<<(std::ostream &os, const Segmentation &seg)
{
  os << "segType " << seg.mSegType << "-" << (seg.mIsBendingPlane ? "B" : "NB");

  os << boost::format(" %3d PG %2d PGT %2d PS\n") % seg.mPadGroups.size() % seg.mPadGroupTypes.size() %
        seg.mPadSizes.size();

  dump(os, "PG", seg.mPadGroups, seg.mPadGroups.size());
  dump(os, "PGT", seg.mPadGroupTypes, seg.mPadGroupTypes.size());
  dump(os, "PS", seg.mPadSizes, seg.mPadSizes.size());
  return os;
}

}
}
}
}

