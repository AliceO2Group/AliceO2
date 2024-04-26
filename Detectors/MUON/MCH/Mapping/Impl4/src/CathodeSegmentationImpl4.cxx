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

///
/// @author  Laurent Aphecetche

#include "CathodeSegmentationImpl4.h"
#include "boost/format.hpp"
#include <boost/geometry.hpp>
#include "GenDetElemId2SegType.h"
#include "PadGroup.h"
#include "PadSize.h"
#include "MCHMappingInterface/CathodeSegmentation.h"
#include "CathodeSegmentationCreator.h"
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <gsl/gsl>
#include <fmt/format.h>

using namespace o2::mch::mapping::impl4;

namespace o2
{
namespace mch
{
namespace mapping
{
namespace impl4
{

CathodeSegmentation* createCathodeSegmentation(int detElemId,
                                               bool isBendingPlane)
{
  int segType = detElemId2SegType(detElemId);
  CathodeSegmentationCreator creator = getCathodeSegmentationCreator(segType);
  if (creator == nullptr) {
    return nullptr;
  }
  return creator(isBendingPlane);
}

void CathodeSegmentation::fillRtree()
{
  int catPadIndex{0};

  for (auto padGroupIndex = 0; padGroupIndex < mPadGroups.size();
       ++padGroupIndex) {
    mPadGroupIndex2CatPadIndexIndex.push_back(catPadIndex);
    auto& pg = mPadGroups[padGroupIndex];
    auto& pgt = mPadGroupTypes[pg.mPadGroupTypeId];
    double dx{mPadSizes[pg.mPadSizeId].first};
    double dy{mPadSizes[pg.mPadSizeId].second};
    for (int ix = 0; ix < pgt.getNofPadsX(); ++ix) {
      for (int iy = 0; iy < pgt.getNofPadsY(); ++iy) {
        if (pgt.id(ix, iy) >= 0) {

          double xmin = ix * dx + pg.mX;
          double xmax = (ix + 1) * dx + pg.mX;
          double ymin = iy * dy + pg.mY;
          double ymax = (iy + 1) * dy + pg.mY;

          mRtree.insert(std::make_pair(
            CathodeSegmentation::Box{CathodeSegmentation::Point(xmin, ymin),
                                     CathodeSegmentation::Point(xmax, ymax)},
            catPadIndex));

          mCatPadIndex2PadGroupIndex.push_back(padGroupIndex);
          mCatPadIndex2PadGroupTypeFastIndex.push_back(pgt.fastIndex(ix, iy));
          ++catPadIndex;
        }
      }
    }
  }
}

std::set<int> getUnique(const std::vector<PadGroup>& padGroups)
{
  // extract from padGroup vector the unique integer values given by func
  std::set<int> u;
  for (auto& pg : padGroups) {
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

CathodeSegmentation::CathodeSegmentation(
  int segType, bool isBendingPlane, std::vector<PadGroup> padGroups,
  std::vector<PadGroupType> padGroupTypes,
  std::vector<std::pair<float, float>> padSizes)
  : mSegType{segType},
    mIsBendingPlane{isBendingPlane},
    mPadGroups{std::move(padGroups)},
    mDualSampaIds{getUnique(mPadGroups)},
    mPadGroupTypes{std::move(padGroupTypes)},
    mPadSizes{std::move(padSizes)},
    mCatPadIndex2PadGroupIndex{},
    mCatPadIndex2PadGroupTypeFastIndex{},
    mPadGroupIndex2CatPadIndexIndex{},
    mDualSampaId2CatPadIndices{}
{
  fillRtree();
  for (auto dualSampaId : mDualSampaIds) {
    mDualSampaId2CatPadIndices.emplace(dualSampaId, catPadIndices(dualSampaId));
  }
}

std::vector<int> CathodeSegmentation::catPadIndices(int dualSampaId) const
{
  std::vector<int> pi;

  for (auto padGroupIndex = 0; padGroupIndex < mPadGroups.size();
       ++padGroupIndex) {
    if (mPadGroups[padGroupIndex].mFECId == dualSampaId) {
      const auto& pgt = mPadGroupTypes[mPadGroups[padGroupIndex].mPadGroupTypeId];
      const auto& i1 = mPadGroupIndex2CatPadIndexIndex[padGroupIndex];
      for (auto i = i1; i < i1 + pgt.getNofPads(); ++i) {
        pi.emplace_back(i);
      }
    }
  }
  return pi;
}

std::vector<int> CathodeSegmentation::getCatPadIndices(int dualSampaId) const
{
  auto it = mDualSampaId2CatPadIndices.find(dualSampaId);
  if (it == mDualSampaId2CatPadIndices.end()) {
    return {};
  }
  return it->second;
}

std::vector<int> CathodeSegmentation::getCatPadIndices(double xmin, double ymin,
                                                       double xmax,
                                                       double ymax) const
{
  std::vector<CathodeSegmentation::Value> result_n;
  mRtree.query(boost::geometry::index::intersects(
                 CathodeSegmentation::Box({xmin, ymin}, {xmax, ymax})),
               std::back_inserter(result_n));
  std::vector<int> catPadIndices;
  for (auto& r : result_n) {
    catPadIndices.push_back(r.second);
  }
  return catPadIndices;
}

std::vector<int> CathodeSegmentation::getNeighbouringCatPadIndices(
  int catPadIndex) const
{
  double x = padPositionX(catPadIndex);
  double y = padPositionY(catPadIndex);
  double dx = padSizeX(catPadIndex) / 2.0;
  double dy = padSizeY(catPadIndex) / 2.0;

  const double offset{0.1}; // 1 mm

  auto pads = getCatPadIndices(x - dx - offset, y - dy - offset, x + dx + offset,
                               y + dy + offset);
  pads.erase(std::remove(begin(pads), end(pads), catPadIndex), end(pads));
  return pads;
}

bool CathodeSegmentation::isValid(int catPadIndex) const
{
  return catPadIndex >= 0 && catPadIndex < static_cast<int>(mCatPadIndex2PadGroupIndex.size());
}

double CathodeSegmentation::squaredDistance(int catPadIndex, double x,
                                            double y) const
{
  double px = padPositionX(catPadIndex) - x;
  double py = padPositionY(catPadIndex) - y;
  return px * px + py * py;
}

int CathodeSegmentation::findPadByPosition(double x, double y) const
{
  const double epsilon{1E-4};
  auto pads =
    getCatPadIndices(x - epsilon, y - epsilon, x + epsilon, y + epsilon);

  double dmin{std::numeric_limits<double>::max()};
  int catPadIndex{InvalidCatPadIndex};

  for (auto i = 0; i < pads.size(); ++i) {
    double d{squaredDistance(pads[i], x, y)};
    if (d < dmin) {
      catPadIndex = pads[i];
      dmin = d;
    }
  }

  return catPadIndex;
}

const PadGroup& CathodeSegmentation::padGroup(int catPadIndex) const
{
  return gsl::at(mPadGroups, mCatPadIndex2PadGroupIndex[catPadIndex]);
}

const PadGroupType& CathodeSegmentation::padGroupType(int catPadIndex) const
{
  return gsl::at(mPadGroupTypes, padGroup(catPadIndex).mPadGroupTypeId);
}

int CathodeSegmentation::findPadByFEE(int dualSampaId,
                                      int dualSampaChannel) const
{
  auto it = mDualSampaId2CatPadIndices.find(dualSampaId);
  if (it == mDualSampaId2CatPadIndices.end()) {
    return InvalidCatPadIndex;
  }
  const auto& padIndices = it->second;
  for (const auto& catPadIndex : padIndices) {
    if (padGroupType(catPadIndex)
          .id(mCatPadIndex2PadGroupTypeFastIndex[catPadIndex]) ==
        dualSampaChannel) {
      return catPadIndex;
    }
  }
  return InvalidCatPadIndex;
}

double CathodeSegmentation::padPositionX(int catPadIndex) const
{
  auto& pg = padGroup(catPadIndex);
  auto& pgt = padGroupType(catPadIndex);
  return pg.mX +
         (pgt.ix(mCatPadIndex2PadGroupTypeFastIndex[catPadIndex]) + 0.5) *
           mPadSizes[pg.mPadSizeId].first;
}

double CathodeSegmentation::padPositionY(int catPadIndex) const
{
  auto& pg = padGroup(catPadIndex);
  auto& pgt = padGroupType(catPadIndex);
  return pg.mY +
         (pgt.iy(mCatPadIndex2PadGroupTypeFastIndex[catPadIndex]) + 0.5) *
           mPadSizes[pg.mPadSizeId].second;
}

double CathodeSegmentation::padSizeX(int catPadIndex) const
{
  return mPadSizes[padGroup(catPadIndex).mPadSizeId].first;
}

double CathodeSegmentation::padSizeY(int catPadIndex) const
{
  return mPadSizes[padGroup(catPadIndex).mPadSizeId].second;
}

int CathodeSegmentation::padDualSampaId(int catPadIndex) const
{
  return padGroup(catPadIndex).mFECId;
}

int CathodeSegmentation::padDualSampaChannel(int catPadIndex) const
{
  return padGroupType(catPadIndex)
    .id(mCatPadIndex2PadGroupTypeFastIndex[catPadIndex]);
}

std::ostream& operator<<(std::ostream& out, const std::pair<float, float>& p)
{
  out << p.first << "," << p.second;
  return out;
}

template <typename T>
void dump(std::ostream& out, const std::string& msg, const std::vector<T>& v,
          int n)
{

  out << msg << "\n";
  for (auto i = 0; i < n; ++i) {
    if (i < v.size()) {
      out << v[i] << "\n";
    }
  }
}

std::ostream& operator<<(std::ostream& os, const CathodeSegmentation& seg)
{
  os << "segType " << seg.mSegType << "-" << (seg.mIsBendingPlane ? "B" : "NB");

  os << boost::format(" %3d PG %2d PGT %2d PS\n") % seg.mPadGroups.size() %
          seg.mPadGroupTypes.size() % seg.mPadSizes.size();

  dump(os, "PG", seg.mPadGroups, seg.mPadGroups.size());
  dump(os, "PGT", seg.mPadGroupTypes, seg.mPadGroupTypes.size());
  dump(os, "PS", seg.mPadSizes, seg.mPadSizes.size());
  return os;
}

} // namespace impl4
} // namespace mapping
} // namespace mch
} // namespace o2
