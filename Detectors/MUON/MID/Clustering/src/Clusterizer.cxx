// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Clustering/src/Clusterizer.cxx
/// \brief  Implementation of the cluster reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   24 October 2016
#include "MIDClustering/Clusterizer.h"

#include <fairlogger/Logger.h>

namespace o2
{
namespace mid
{

//______________________________________________________________________________
bool Clusterizer::loadPreClusters(gsl::span<const PreCluster>& preClusters)
{
  /// Fills the structure with pre-clusters

  // Loop on pre-clusters in the BP
  PreClustersDE* de = nullptr;
  mPreClusters = &preClusters;
  for (auto& pc : preClusters) {
    int deIndex = pc.deId;
    de = &mPreClustersDE[deIndex];
    de->setDEId(deIndex);
    size_t idx = &pc - &preClusters[0];
    if (pc.cathode == 0) {
      de->getPreClustersBP(pc.firstColumn).push_back({ idx, 0, mPreClusterHelper.getArea(pc) });
    } else {
      de->getPreClustersNBP().push_back({ idx, 0 });
      for (int icolumn = pc.firstColumn; icolumn <= pc.lastColumn; ++icolumn) {
        de->getPreClustersNBP().back().area[icolumn] = mPreClusterHelper.getArea(icolumn, pc);
      }
    }

    mActiveDEs[deIndex] = true;
  }

  return (preClusters.size() > 0);
}

//______________________________________________________________________________
bool Clusterizer::process(gsl::span<const PreCluster> preClusters)
{
  /// Main function: runs on the preclusters and builds the clusters
  /// \param preClusters gsl::span of PreClusters objects in the

  // Reset cluster information
  reset();
  if (loadPreClusters(preClusters)) {
    // Loop only on fired detection elements
    for (auto& deIndex : mActiveDEs) {
      makeClusters(mPreClustersDE[deIndex.first]);
    }
    return true;
  }

  return false;
}

//______________________________________________________________________________
bool Clusterizer::makeClusters(PreClustersDE& pcs)
{
  /// Makes the clusters and stores it
  int deIndex = pcs.getDEId();

  LOG(DEBUG) << "Clusterizing " << deIndex;

  // loop over pre-clusters in the non-bending plane
  for (int inb = 0; inb < pcs.getNPreClustersNBP(); ++inb) {
    auto& pcNB = pcs.getPreClusterNBP(inb);
    // Try to find matching pre-clusters in the bending plane
    int icolumn = (*mPreClusters)[pcNB.index].firstColumn;
    // bool isNBPpaired = false;
    if (icolumn == (*mPreClusters)[pcNB.index].lastColumn) {
      // This is the most simple and common case: the NBP pre-cluster is on
      // one single column. So it can be easily matched with the BP
      // since the corresponding contours are both rectangles
      for (int ib = 0; ib < pcs.getNPreClustersBP(icolumn); ++ib) {
        auto& pcB = pcs.getPreClusterBP(icolumn, ib);
        makeCluster(pcB.area, pcNB.area[icolumn], icolumn, deIndex);
        pcB.paired = 1;
        pcNB.paired = 1;
        mFunction(mClusters.size() - 1, pcB.index);
        mFunction(mClusters.size() - 1, pcNB.index);
        // isNBPpaired = true;
        // setPairedFlag(icolumn, ib);
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
      LOG(DEBUG) << "Spanning non-bend: " << icolumn << " -> " << (*mPreClusters)[pcNB.index].lastColumn;
      int pairId = 10 + inb;
      for (; icolumn <= (*mPreClusters)[pcNB.index].lastColumn; ++icolumn) {
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
            makeCluster(pcB.area, pcNB.area[icolumn], icolumn, deIndex);
            mFunction(mClusters.size() - 1, pcB.index);
            mFunction(mClusters.size() - 1, pcNB.index);
          } else {
            for (auto& jb : neighbours) {
              PreClustersDE::BP& pcBneigh = pcs.getPreClusterBP(neighColumn, jb);
              makeCluster(pcB, pcBneigh, pcNB, deIndex);
              // Here we set the paired flag to a custom ID of the pre-cluster in the NBP
              // So that, when we move to the next column, we do not add it twice.
              pcBneigh.paired = pairId;
              // setPairedFlag(neighColumn, jb, pairId);
              mFunction(mClusters.size() - 1, pcB.index);
              mFunction(mClusters.size() - 1, pcBneigh.index);
              mFunction(mClusters.size() - 1, pcNB.index);
            }
          }
          pcB.paired = 1;
          pcNB.paired = 1;
          // isNBPpaired = true;
          // setPairedFlag(icolumn, ib);
        } // loop on pre-clusters in the BP
      }   // loop on columns
    }

    if (pcNB.paired == 0) {
      // If it is not paired, it means that we have
      // a monocathodic cluster in the NBP
      for (int icolumn = (*mPreClusters)[pcNB.index].firstColumn; icolumn <= (*mPreClusters)[pcNB.index].lastColumn; ++icolumn) {
        makeCluster(pcNB.area[icolumn], pcNB.area[icolumn], icolumn, deIndex);
        mFunction(mClusters.size() - 1, pcNB.index);
      }
    }
  } // loop on pre-clusters in the NBP

  /// Search for monocathodic clusters in the BP
  for (int icolumn = 0; icolumn <= 6; ++icolumn) {
    for (int ib = 0; ib < pcs.getNPreClustersBP(icolumn); ++ib) {
      PreClustersDE::BP& pcB = pcs.getPreClusterBP(icolumn, ib);
      if (pcB.paired == 0) {
        makeCluster(pcB.area, pcB.area, icolumn, deIndex);
        mFunction(mClusters.size() - 1, pcB.index);
      }
      // Reset the value
      // mPairedFlag[icolumn][ib] = 0;
    }
  }

  // Reset the pre-clusters before exiting
  pcs.reset();

  return true;
}

//______________________________________________________________________________
bool Clusterizer::init(std::function<void(size_t, size_t)> func)
{
  /// Initializes the class

  // prepare storage of clusters and PreClusters
  mClusters.reserve(100);
  mPreClustersDE.reserve(72);
  mFunction = func;
  // for (auto& paired : mPairedFlag) {
  //   paired.reserve(10);
  // }

  return true;
}

//______________________________________________________________________________
void Clusterizer::makeCluster(const MpArea& areaBP, const MpArea& areaNBP, const int& icolumn, const int& deIndex)
{
  /// Makes the cluster from pre-clusters
  float xCoor = 0.5 * (areaNBP.getXmax() + areaNBP.getXmin());
  float yCoor = 0.5 * (areaBP.getYmax() + areaBP.getYmin());
  double deltaX = areaNBP.getXmax() - areaNBP.getXmin();
  double deltaY = areaBP.getYmax() - areaBP.getYmin();
  float sigmaX2 = deltaX * deltaX / 12;
  float sigmaY2 = deltaY * deltaY / 12;
  mClusters.push_back({ static_cast<uint8_t>(deIndex), xCoor, yCoor, sigmaX2, sigmaY2 });

  LOG(DEBUG) << "pos: (" << xCoor << ", " << yCoor << ") err2: (" << sigmaX2 << ", " << sigmaY2 << ")";
}

//______________________________________________________________________________
void Clusterizer::makeCluster(const PreClustersDE::BP& pcBP, const PreClustersDE::BP& pcBPNeigh, const PreClustersDE::NBP& pcNBP, const int& deIndex)
{
  /// Makes the cluster from pre-clusters
  // This is the general case:
  // perform the full calculation assuming a uniform charge distribution

  double x2[2][2] = { { 0., 0. }, { 0., 0. } };
  double x3[2][2] = { { 0., 0. }, { 0., 0. } };
  double dim[2][2];
  double delta[2];
  double sumArea = 0.;

  std::vector<const PreClustersDE::BP*> pcBlist = { &pcBP, &pcBPNeigh };

  for (auto* pc : pcBlist) {
    int icolumn = (*mPreClusters)[pc->index].firstColumn;
    dim[0][0] = pcNBP.area[icolumn].getXmin();
    dim[0][1] = pcNBP.area[icolumn].getXmax();
    dim[1][0] = pc->area.getYmin();
    dim[1][1] = pc->area.getYmax();
    for (int iplane = 0; iplane < 2; ++iplane) {
      delta[iplane] = dim[iplane][1] - dim[iplane][0];
    }
    // area = dx * dy
    sumArea += delta[0] * delta[1];
    LOG(DEBUG) << "Area += " << delta[0] * delta[1] << " => " << sumArea;
    for (int iplane = 0; iplane < 2; ++iplane) {
      for (int ip = 0; ip < 2; ++ip) {
        // second momentum = x_i * x_i * dy
        double currX2 = dim[iplane][ip] * dim[iplane][ip] * delta[1 - iplane];
        x2[iplane][ip] += currX2;
        // third momentum = x_i * x_i * x_i * dy
        x3[iplane][ip] += currX2 * dim[iplane][ip];
        LOG(DEBUG) << "x[" << iplane << "][" << ip << "] => val " << dim[iplane][ip] << " delta " << delta[1 - iplane]
                   << " => x2 " << x2[iplane][ip] << " x3 " << x3[iplane][ip];
      }
    }
  } // loop on column

  double coor[2], sigma2[2];
  for (int iplane = 0; iplane < 2; ++iplane) {
    coor[iplane] = (x2[iplane][1] - x2[iplane][0]) / sumArea / 2.;
    sigma2[iplane] = (x3[iplane][1] - x3[iplane][0]) / sumArea / 3. - coor[iplane] * coor[iplane];
  }

  mClusters.push_back({ static_cast<uint8_t>(deIndex), static_cast<float>(coor[0]), static_cast<float>(coor[1]), static_cast<float>(sigma2[0]), static_cast<float>(sigma2[1]) });

  LOG(DEBUG) << "pos: (" << coor[0] << ", " << coor[1] << ") err2: (" << sigma2[0] << ", " << sigma2[1] << ")";
}

//______________________________________________________________________________
void Clusterizer::reset()
{
  /// Resets the clusters
  mActiveDEs.clear();
  mClusters.clear();
}
} // namespace mid
} // namespace o2
