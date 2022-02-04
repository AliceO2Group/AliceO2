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

/// \file   MID/Clustering/src/Clusterizer.cxx
/// \brief  Implementation of the cluster reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   24 October 2016
#include "MIDClustering/Clusterizer.h"

#include "MIDBase/DetectorParameters.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
bool Clusterizer::loadPreClusters(gsl::span<const PreCluster>& preClusters)
{
  // Loop on pre-clusters in the BP
  mActiveDEs.clear();
  mPreClusters = preClusters;
  for (auto& pc : preClusters) {
    auto& de = mPreClustersDE[pc.deId];
    de.setDEId(pc.deId);
    size_t idx = &pc - &preClusters[0];
    if (pc.cathode == 0) {
      de.getPreClustersBP(pc.firstColumn).push_back({idx, 0, mPreClusterHelper.getArea(pc)});
    } else {
      de.getPreClustersNBP().push_back({idx, 0});
      for (int icolumn = pc.firstColumn; icolumn <= pc.lastColumn; ++icolumn) {
        de.getPreClustersNBP().back().area[icolumn] = mPreClusterHelper.getArea(icolumn, pc);
      }
    }

    mActiveDEs.emplace(pc.deId);
  }

  return (preClusters.size() > 0);
}

//______________________________________________________________________________
void Clusterizer::process(gsl::span<const PreCluster> preClusters, bool accumulate)
{
  // Reset cluster information
  if (!accumulate) {
    mClusters.clear();
  }
  if (loadPreClusters(preClusters)) {
    // Loop only on fired detection elements
    for (auto& deId : mActiveDEs) {
      makeClusters(mPreClustersDE[deId]);
    }
  }
}

void Clusterizer::process(gsl::span<const PreCluster> preClusters, gsl::span<const ROFRecord> rofRecords)
{
  mClusters.clear();
  mROFRecords.clear();
  for (auto& rofRecord : rofRecords) {
    mPreClusterOffset = rofRecord.firstEntry;
    auto firstEntry = mClusters.size();
    process(preClusters.subspan(rofRecord.firstEntry, rofRecord.nEntries), true);
    auto nEntries = mClusters.size() - firstEntry;
    mROFRecords.emplace_back(rofRecord, firstEntry, nEntries);
  }
}

//______________________________________________________________________________
bool Clusterizer::makeClusters(PreClustersDE& pcs)
{
  auto deId = pcs.getDEId();

  // loop over pre-clusters in the non-bending plane
  for (int inb = 0; inb < pcs.getNPreClustersNBP(); ++inb) {
    auto& pcNB = pcs.getPreClusterNBP(inb);
    // Try to find matching pre-clusters in the bending plane
    int icolumn = mPreClusters[pcNB.index].firstColumn;
    if (icolumn == mPreClusters[pcNB.index].lastColumn) {
      // This is the most simple and common case: the NBP pre-cluster is on
      // one single column. So it can be easily matched with the BP
      // since the corresponding contours are both rectangles
      for (int ib = 0; ib < pcs.getNPreClustersBP(icolumn); ++ib) {
        auto& pcB = pcs.getPreClusterBP(icolumn, ib);
        makeCluster(pcB.area, pcNB.area[icolumn], deId);
        pcB.paired = 1;
        pcNB.paired = 1;
        mFunction(mClusters.size() - 1, pcB.index + mPreClusterOffset);
        mFunction(mClusters.size() - 1, pcNB.index + mPreClusterOffset);
      }
    } else {
      // The NBP pre-cluster spans different columns.
      // The BP contour is therefore a series of neighbour rectangles
      // Here we could build a giant cluster with all neighbour pre-clusters
      // in the bending plane.
      // However, the pre-cluster in the BP might be close to more than one pre-clusters
      // in the BP of the neighbour column, thus leading to a pattern with non-contiguous
      // pre-clusters along the y axis...which are treated as separate clusters
      // in the standard configuration.
      // All in all, since the strips are huge, we decided to pair the clusters only between
      // two adjacent columns.
      // In this way, a huge cluster spanning three columns will be transformed in
      // two clusters centered at the border between two columns.
      // This leads to a redundancy. Still, the sigma of the resulting cluster will be so large
      // to have no impact whatsoever in the track reconstruction.
      int pairId = 10 + inb;
      for (; icolumn <= mPreClusters[pcNB.index].lastColumn; ++icolumn) {
        int neighColumn = icolumn + 1;
        for (int ib = 0; ib < pcs.getNPreClustersBP(icolumn); ++ib) {
          PreClustersDE::BP& pcB = pcs.getPreClusterBP(icolumn, ib);
          // This function checks for the neighbours only in icolumn+1
          std::vector<int> neighbours = pcs.getNeighbours(icolumn, ib);
          // It can happen that the NBP spans two columns...but there are fired strips only on
          // one column of the BP. In this case we only consider the column with a hit in the BP.
          // Of course, we need to check that the current pre-cluster was not already paired
          // with the pre-cluster in the previous column.
          if (neighbours.empty() && pcB.paired != pairId) {
            makeCluster(pcB.area, pcNB.area[icolumn], deId);
            mFunction(mClusters.size() - 1, pcB.index + mPreClusterOffset);
            mFunction(mClusters.size() - 1, pcNB.index + mPreClusterOffset);
          } else {
            for (auto& jb : neighbours) {
              PreClustersDE::BP& pcBneigh = pcs.getPreClusterBP(neighColumn, jb);
              makeCluster(pcB, pcBneigh, pcNB, deId);
              // Here we set the paired flag to a custom ID of the pre-cluster in the NBP
              // So that, when we move to the next column, we do not add it twice.
              pcBneigh.paired = pairId;
              mFunction(mClusters.size() - 1, pcB.index + mPreClusterOffset);
              mFunction(mClusters.size() - 1, pcBneigh.index + mPreClusterOffset);
              mFunction(mClusters.size() - 1, pcNB.index + mPreClusterOffset);
            }
          }
          pcB.paired = 1;
          pcNB.paired = 1;
        } // loop on pre-clusters in the BP
      }   // loop on columns
    }

    if (pcNB.paired == 0) {
      // If it is not paired, it means that we have
      // a monocathodic cluster in the NBP
      for (int icolumn = mPreClusters[pcNB.index].firstColumn; icolumn <= mPreClusters[pcNB.index].lastColumn; ++icolumn) {
        makeCluster(pcNB.area[icolumn], deId, 1);
        mFunction(mClusters.size() - 1, pcNB.index + mPreClusterOffset);
      }
    }
  } // loop on pre-clusters in the NBP

  /// Search for monocathodic clusters in the BP
  for (int icolumn = 0; icolumn <= 6; ++icolumn) {
    for (int ib = 0; ib < pcs.getNPreClustersBP(icolumn); ++ib) {
      PreClustersDE::BP& pcB = pcs.getPreClusterBP(icolumn, ib);
      if (pcB.paired == 0) {
        makeCluster(pcB.area, deId, 0);
        mFunction(mClusters.size() - 1, pcB.index + mPreClusterOffset);
      }
    }
  }

  // Reset the pre-clusters before exiting
  pcs.reset();

  return true;
}

//______________________________________________________________________________
bool Clusterizer::init(std::function<void(size_t, size_t)> func)
{
  // prepare storage of clusters and PreClusters
  mPreClustersDE.reserve(detparams::NDetectionElements);
  mFunction = func;

  return true;
}

//______________________________________________________________________________
void Clusterizer::makeCluster(const MpArea& areaBP, const MpArea& areaNBP, uint8_t deId)
{
  constexpr double sqrt12 = 3.4641016;
  double xCoor = 0.5 * (areaNBP.getXmax() + areaNBP.getXmin());
  double yCoor = 0.5 * (areaBP.getYmax() + areaBP.getYmin());
  double errX = (areaNBP.getXmax() - areaNBP.getXmin()) / sqrt12;
  double errY = (areaBP.getYmax() - areaBP.getYmin()) / sqrt12;
  mClusters.push_back({static_cast<float>(xCoor), static_cast<float>(yCoor), 0., static_cast<float>(errX), static_cast<float>(errY), deId});
  mClusters.back().setBothFired();
}

//______________________________________________________________________________
void Clusterizer::makeCluster(const MpArea& area, uint8_t deId, int cathode)
{
  makeCluster(area, area, deId);
  mClusters.back().setFired(1 - cathode, false);
}

//______________________________________________________________________________
void Clusterizer::makeCluster(const PreClustersDE::BP& pcBP, const PreClustersDE::BP& pcBPNeigh, const PreClustersDE::NBP& pcNBP, uint8_t deId)
{
  // This is the general case:
  // perform the full calculation assuming a uniform charge distribution

  double x2[2][2] = {{0., 0.}, {0., 0.}};
  double x3[2][2] = {{0., 0.}, {0., 0.}};
  double dim[2][2];
  double delta[2];
  double sumArea = 0.;

  std::vector<const PreClustersDE::BP*> pcBlist = {&pcBP, &pcBPNeigh};

  for (auto* pc : pcBlist) {
    int icolumn = mPreClusters[pc->index].firstColumn;
    dim[0][0] = pcNBP.area[icolumn].getXmin();
    dim[0][1] = pcNBP.area[icolumn].getXmax();
    dim[1][0] = pc->area.getYmin();
    dim[1][1] = pc->area.getYmax();
    for (int iplane = 0; iplane < 2; ++iplane) {
      delta[iplane] = dim[iplane][1] - dim[iplane][0];
    }
    // area = dx * dy
    sumArea += delta[0] * delta[1];
    for (int iplane = 0; iplane < 2; ++iplane) {
      for (int ip = 0; ip < 2; ++ip) {
        // second momentum = x_i * x_i * dy
        double currX2 = dim[iplane][ip] * dim[iplane][ip] * delta[1 - iplane];
        x2[iplane][ip] += currX2;
        // third momentum = x_i * x_i * x_i * dy
        x3[iplane][ip] += currX2 * dim[iplane][ip];
      }
    }
  } // loop on column

  double coor[2], sigma[2];
  for (int iplane = 0; iplane < 2; ++iplane) {
    coor[iplane] = (x2[iplane][1] - x2[iplane][0]) / sumArea / 2.;
    sigma[iplane] = std::sqrt((x3[iplane][1] - x3[iplane][0]) / sumArea / 3. - coor[iplane] * coor[iplane]);
  }

  mClusters.push_back({static_cast<float>(coor[0]), static_cast<float>(coor[1]), 0., static_cast<float>(sigma[0]), static_cast<float>(sigma[1]), deId});
  mClusters.back().setBothFired();
}

//______________________________________________________________________________
void Clusterizer::reset()
{
  mActiveDEs.clear();
  mClusters.clear();
}
} // namespace mid
} // namespace o2
