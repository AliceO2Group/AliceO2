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

/// \file  testO2TPCIDCAverageGroup.cxx
/// \brief this task tests grouping and averaging
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#define BOOST_TEST_MODULE Test TPC O2TPCIDCAverageGroup class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCCalibration/IDCAverageGroup.h"
#include <vector>
#include <algorithm>

namespace o2
{
namespace tpc
{

static constexpr unsigned short MAXGRPAD = 10;           // maximum number of pads grouped
static constexpr unsigned short MAXGRROW = 4;            // maximum number of pads grouped
static constexpr unsigned short MAXGRTHR = 2;            // maximum number of pads grouped
static constexpr unsigned int MAXGRSECEDGE = 4;          // maximum number of pads grouped
static constexpr unsigned int NINTEGRATIONINTERVALS = 3; // maximum number of pads grouped
static constexpr const int MAXGROUPS = 4;                // maximum number of pads at the sector edges which are grouped
static constexpr const int NITER = 2;                    // number of iterations performed for the testing

unsigned int genRand(const int maxVal)
{
  return rand() % maxVal;
}

unsigned int genRandSecEdge()
{
  unsigned int groupPadsSectorEdges = genRand(MAXGRSECEDGE);
  for (int i = 0; i < MAXGROUPS - 1; ++i) {
    groupPadsSectorEdges *= 10;
    groupPadsSectorEdges += genRand(MAXGRSECEDGE) + 1;
  }
  return groupPadsSectorEdges;
}

std::vector<float> getIDCs(const int region)
{
  const int nIDCs = Mapper::PADSPERREGION[region];
  std::vector<float> idcsTmp(nIDCs);
  std::iota(std::begin(idcsTmp), std::end(idcsTmp), 0);

  std::vector<float> idcs;
  for (unsigned int i = 0; i < NINTEGRATIONINTERVALS; ++i) {
    idcs.insert(idcs.end(), idcsTmp.begin(), idcsTmp.end());
  }
  return idcs;
}

std::vector<float> getIDCsSide()
{
  std::vector<float> idcs;

  std::array<std::vector<float>, Mapper::NREGIONS> idcsPerRegion;
  for (int region = 0; region < Mapper::NREGIONS; ++region) {
    const int nIDCs = Mapper::PADSPERREGION[region];
    auto& idc = idcsPerRegion[region];
    idc.resize(nIDCs);
    std::iota(std::begin(idc), std::end(idc), 0);
  }

  for (unsigned int i = 0; i < NINTEGRATIONINTERVALS; ++i) {
    for (int sector = 0; sector < SECTORSPERSIDE; ++sector) {
      for (int region = 0; region < Mapper::NREGIONS; ++region) {
        const auto& idc = idcsPerRegion[region];
        idcs.insert(idcs.end(), idc.begin(), idc.end());
      }
    }
  }
  return idcs;
}

BOOST_AUTO_TEST_CASE(AverageGroupSector_test)
{
  std::srand(std::time(nullptr));

  const static auto& paramIDCGroup = ParameterIDCGroup::Instance();
  BOOST_CHECK(paramIDCGroup.method == AveragingMethod::FAST);

  // create some random IDCs for one TPC side
  const auto idcsPerSide = getIDCsSide();

  std::array<std::vector<float>, Mapper::NREGIONS> idcsRegion;
  for (int i = 0; i < Mapper::NREGIONS; ++i) {
    idcsRegion[i] = getIDCs(i);
  }

  // get the total sum of the reference IDCs (consider rounding of floats)
  unsigned long refSumIDCs = 0;
  for (const auto val : idcsPerSide) {
    refSumIDCs += static_cast<unsigned long>(val + 0.1f);
  }
  refSumIDCs *= 2;

  // converting the IDCs to IDCDelta object which is used forthe averating and grouping
  IDCDelta<float> idcs;
  idcs.getIDCDelta() = idcsPerSide;

  for (int iter = 0; iter < NITER; ++iter) {
    // type=0: do nothing special for grouping at sector edges, typ1=1: do group pads at sector edges in pad direction, typ1=2: group pads at sector edges in pad+row direction
    for (int type = 0; type < 3; ++type) {
      const unsigned char grPadTmp = genRand(MAXGRPAD) + 1;
      const unsigned char grRowTmp = genRand(MAXGRROW) + 1;
      const unsigned char grRowThrTmp = genRand(MAXGRTHR);
      const unsigned char grPadThrTmp = genRand(MAXGRTHR);
      const unsigned int groupPadsSectorEdges = type == 0 ? 0 : 10 * genRandSecEdge() + type - 1;
      std::array<unsigned char, Mapper::NREGIONS> grPad{};
      std::fill(grPad.begin(), grPad.end(), grPadTmp);
      std::array<unsigned char, Mapper::NREGIONS> grRow{};
      std::fill(grRow.begin(), grRow.end(), grRowTmp);
      std::array<unsigned char, Mapper::NREGIONS> grRowThr{};
      std::fill(grRowThr.begin(), grRowThr.end(), grRowThrTmp);
      std::array<unsigned char, Mapper::NREGIONS> grPadThr{};
      std::fill(grPadThr.begin(), grPadThr.end(), grPadThrTmp);

      // perform the grouping
      IDCAverageGroup<IDCAverageGroupTPC> idcaverage(grPad, grRow, grRowThr, grPadThr, groupPadsSectorEdges);
      idcaverage.setIDCs(idcs, Side::A);
      idcaverage.processIDCs();

      std::array<IDCAverageGroup<IDCAverageGroupCRU>, Mapper::NREGIONS> idcaverageCRU{
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 0),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 1),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 2),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 3),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 4),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 5),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 6),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 7),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 8),
        IDCAverageGroup<IDCAverageGroupCRU>(grPadTmp, grRowTmp, grRowThrTmp, grPadThrTmp, groupPadsSectorEdges, 9)};

      for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
        idcaverageCRU[region].setIDCs(idcsRegion[region]);
        idcaverageCRU[region].processIDCs();
      }

      // get the total sum of the grouped IDCs (consider rounding of floats)
      unsigned long sumGroupedIDCs = 0;
      unsigned long sumGroupedIDCsCRU = 0;
      std::vector<unsigned long> sumGroupedIDCsSector(NINTEGRATIONINTERVALS * Mapper::NSECTORS);
      for (unsigned int i = 0; i < NINTEGRATIONINTERVALS; ++i) {
        for (unsigned int sector = 0; sector < Mapper::NSECTORS; ++sector) {
          const int index = sector + i * Mapper::NSECTORS;
          for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
            for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
              for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
                const auto valSector = static_cast<unsigned long>(idcaverage.getUngroupedIDCDeltaVal(sector, region, irow, ipad, i) + 0.1f);
                const auto valCRU = static_cast<unsigned long>(idcaverageCRU[region].getUngroupedIDCValLocal(irow, ipad, i) + 0.1f);
                sumGroupedIDCsSector[index] += valSector;
                sumGroupedIDCsCRU += valCRU;
                BOOST_CHECK(valSector == valCRU);
              }
            }
          }
          sumGroupedIDCs += sumGroupedIDCsSector[i];
        }
      }

      // comparing the sum from the reference IDCs to the sum of the grouped IDCs
      BOOST_CHECK(refSumIDCs == sumGroupedIDCs);
      BOOST_CHECK(refSumIDCs == sumGroupedIDCsCRU);

      // comparing the reference sum of IDCs per sector with the grouped IDCs per sector
      for (const auto idcsSec : sumGroupedIDCsSector) {
        BOOST_CHECK(refSumIDCs / sumGroupedIDCsSector.size() == idcsSec);
      }
    }
  }
}

} // namespace tpc
} // namespace o2
