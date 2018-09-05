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
#include <cassert>

#include <fairlogger/Logger.h>

namespace o2
{
namespace mid
{

//______________________________________________________________________________
Clusterizer::Clusterizer()
  : mClusters()
{
  /// Default constructor
}

//______________________________________________________________________________
bool Clusterizer::process(std::vector<PreClusters>& preClusters)
{
  /// Main function: runs on the preclusters and builds the clusters
  /// @param preClusters Vector of PreClusters objects
  // Reset cluster information
  reset();
  for (auto& pcs : preClusters) {
    makeClusters(pcs);
  }
  return true;
}

//______________________________________________________________________________
bool Clusterizer::makeClusters(PreClusters& pcs)
{
  /// Makes the clusters and stores it
  int deIndex = pcs.getDEId();

  LOG(DEBUG) << "Clusterizing " << deIndex;

  // loop over pre-clusters in the non-bending plane
  for (int inb = 0; inb < pcs.getNPreClustersNBP(); ++inb) {
    PreClusters::PreClusterNBP& pcNB = pcs.getPreClusterNBP(inb);
    // Try to find matching pre-clusters in the bending plane
    int icolumn = pcNB.firstColumn;
    if (pcNB.firstColumn == pcNB.lastColumn) {
      // This is the most simple and common case: the NBP pre-cluster is on
      // one single column. So it can be easily matched with the BP
      // since the corresponding contours are both rectangles
      for (int ib = 0; ib < pcs.getNPreClustersBP(icolumn); ++ib) {
        makeCluster(pcs.getPreClusterBP(icolumn, ib), pcNB, deIndex);
      }
    } else {
      // The NBP pre-cluster spans different columns.
      // The BP contour is therefore a serie of neighbour rectangles
      // Here we could build a giant cluster with all neighbour pre-clusters
      // in the bending plane.
      // However, the a pre-cluster in the BP might be close to more than one pre-clusters
      // in the BP of the close column, thus leading to a pattern with non-contiguous
      // pre-clusters along the y axis...which are treated as separate clusters
      // in the standard configuration.
      // All in all, since the strips are huge, we decided to pair the clusters only between
      // two adjacent columns.
      // In this way, a huge cluster spanning three columns will be transformed in
      // two clusters centered at the border between two columns.
      // This leads to a redundancy. Still, the sigma of the resulting cluster will be so large
      // to have no impact whatsoever in the track reconstruction.
      LOG(DEBUG) << "Spanning non-bend: " << icolumn << " -> " << pcNB.lastColumn;
      int pairId = 10 + inb;
      for (; icolumn <= pcNB.lastColumn; ++icolumn) {
        for (int ib = 0; ib < pcs.getNPreClustersBP(icolumn); ++ib) {
          PreClusters::PreClusterBP& pcB = pcs.getPreClusterBP(icolumn, ib);
          // This function checks for the neighbours only on icolumn+1
          std::vector<int> neighbours = pcs.getNeighbours(icolumn, ib);
          // It can happen that the NBP spans two columns...but there are fired strips only on
          // one column of the BP. In this case we only consider the column with a hit in the BP.
          // Of course, we need to check that the current pre-cluster was not already paired
          // with the pre-cluster in the previous column.
          if (neighbours.empty() && pcB.paired != pairId) {
            makeCluster(pcB, pcNB, deIndex);
          } else {
            for (auto& jb : neighbours) {
              PreClusters::PreClusterBP& pcBneigh = pcs.getPreClusterBP(icolumn + 1, jb);
              makeCluster(pcB, pcBneigh, pcNB, deIndex);
              // Here we set the paired flag to a custom ID of the pre-cluster in the NBP
              // So that, when we move to the next column, we do not add it twice.
              pcBneigh.paired = pairId;
            }
          }
        } // loop on pre-clusters in the BP
      }   // loop on columns
    }

    if (pcNB.paired == 0) {
      // If it is not paired, it means that we have
      // a monocathodic cluster in the NBP
      makeCluster(pcNB, deIndex);
    }
  } // loop on pre-clusters in the NBP

  /// Search for monocathodic clusters in the BP
  for (int icolumn = 0; icolumn <= 6; ++icolumn) {
    for (int ib = 0; ib < pcs.getNPreClustersBP(icolumn); ++ib) {
      PreClusters::PreClusterBP& pcB = pcs.getPreClusterBP(icolumn, ib);
      if (pcB.paired == 0) {
        makeCluster(pcB, deIndex);
      }
    }
  }

  return true;
}

//______________________________________________________________________________
bool Clusterizer::init()
{
  /// Initializes the class

  // prepare storage of clusters and PreClusters
  mClusters.reserve(100);

  return true;
}

//______________________________________________________________________________
Cluster2D& Clusterizer::nextCluster()
{
  /// Iterates on clusters
  if (mNClusters >= static_cast<uint32_t>(mClusters.size())) {
    mClusters.emplace_back(Cluster2D());
  }
  Cluster2D& cl = mClusters[mNClusters];
  ++mNClusters;
  return cl;
}

//______________________________________________________________________________
void Clusterizer::makeCluster(PreClusters::PreClusterBP& clBend, PreClusters::PreClusterNBP& clNonBend, const int& deIndex)
{
  /// Makes the cluster from pre-clusters
  Cluster2D& cl = nextCluster();
  int icolumn = clBend.column;
  cl.deId = (uint8_t)deIndex;
  cl.xCoor = 0.5 * (clNonBend.area[icolumn].getXmax() + clNonBend.area[icolumn].getXmin());
  cl.yCoor = 0.5 * (clBend.area.getYmax() + clBend.area.getYmin());
  double deltaX = clNonBend.area[icolumn].getXmax() - clNonBend.area[icolumn].getXmin();
  double deltaY = clBend.area.getYmax() - clBend.area.getYmin();
  cl.sigmaX2 = deltaX * deltaX / 12;
  cl.sigmaY2 = deltaY * deltaY / 12;
  clBend.paired = 1;
  clNonBend.paired = 1;

  LOG(DEBUG) << "pos: (" << cl.xCoor << ", " << cl.yCoor << ") err2: (" << cl.sigmaX2 << ", " << cl.sigmaY2 << ")";
}

//______________________________________________________________________________
void Clusterizer::makeCluster(PreClusters::PreClusterNBP& clNonBend, const int& deIndex)
{
  /// Makes the cluster from pre-clusters
  PreClusters::PreClusterBP clBend;
  clBend.column = clNonBend.lastColumn;
  clBend.area = clNonBend.area[clBend.column];
  makeCluster(clBend, clNonBend, deIndex);
}

//______________________________________________________________________________
void Clusterizer::makeCluster(PreClusters::PreClusterBP& clBend, const int& deIndex)
{
  /// Makes the cluster from pre-clusters
  PreClusters::PreClusterNBP clNonBend;
  clNonBend.firstColumn = clNonBend.lastColumn = clBend.column;
  clNonBend.area[clBend.column] = clBend.area;
  makeCluster(clBend, clNonBend, deIndex);
}

//______________________________________________________________________________
void Clusterizer::makeCluster(PreClusters::PreClusterBP& clBend, PreClusters::PreClusterBP& clBendNeigh, PreClusters::PreClusterNBP& clNonBend, const int& deIndex)
{
  /// Makes the cluster from pre-clusters
  // This is the general case:
  // perform the full calculation assuming a uniform charge distribution

  Cluster2D& cl = nextCluster();
  cl.deId = (uint8_t)deIndex;

  double x2[2][2] = { { 0., 0. }, { 0., 0. } };
  double x3[2][2] = { { 0., 0. }, { 0., 0. } };
  double dim[2][2];
  double delta[2];
  double sumArea = 0.;

  std::vector<PreClusters::PreClusterBP*> pcBlist = { &clBend, &clBendNeigh };

  for (auto* pcBP : pcBlist) {
    int icolumn = pcBP->column;
    dim[0][0] = clNonBend.area[icolumn].getXmin();
    dim[0][1] = clNonBend.area[icolumn].getXmax();
    dim[1][0] = pcBP->area.getYmin();
    dim[1][1] = pcBP->area.getYmax();
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
    pcBP->paired = 1;
    clNonBend.paired = 1;
  } // loop on column

  double coor[2], sigma2[2];
  for (int iplane = 0; iplane < 2; ++iplane) {
    coor[iplane] = (x2[iplane][1] - x2[iplane][0]) / sumArea / 2.;
    sigma2[iplane] = (x3[iplane][1] - x3[iplane][0]) / sumArea / 3. - coor[iplane] * coor[iplane];
  }

  cl.xCoor = (float)coor[0];
  cl.yCoor = (float)coor[1];
  cl.sigmaX2 = (float)sigma2[0];
  cl.sigmaY2 = (float)sigma2[1];

  LOG(DEBUG) << "pos: (" << cl.xCoor << ", " << cl.yCoor << ") err2: (" << cl.sigmaX2 << ", " << cl.sigmaY2 << ")";
}

// //______________________________________________________________________________
// bool Clusterizer::buildListOfNeighbours(int icolumn, int lastColumn, std::vector<std::vector<PreCluster*>>& neighbours,
//                                         bool skipPaired, int currentList)
// {
//   /// Build list of neighbours
//   LOG(DEBUG) << "Building list of neighbours in (" << icolumn << ", " << lastColumn << ")";
//   for (int jcolumn = icolumn; jcolumn <= lastColumn; ++jcolumn) {
//     for (int ib = 0; ib < mNPreClusters[jcolumn]; ++ib) {
//       PreCluster* pcB = &mPreClusters[jcolumn][ib];
//       if (skipPaired && pcB->paired > 0) {
//         LOG(DEBUG) << "Column " << jcolumn << "  ib " << ib << "  is already paired => skipPaired";
//         continue;
//       }
//       if (currentList >= neighbours.size()) {
//         // We are starting a new series of neighbour
//         // Let's make sure the pre-cluster is not already part of another list
//         if (pcB->paired == 2) {
//           LOG(DEBUG) << "Column " << jcolumn << "  ib " << ib << "  is already in a list";
//           continue;
//         }
//         LOG(DEBUG) << "New list " << currentList;
//         neighbours.emplace_back(std::vector<PreCluster*>());
//       }
//       std::vector<PreCluster*>& neighList = neighbours[currentList];
//       if (!neighList.empty()) {
//         auto* neigh = neighList.back();
//         // Check if the pre-cluster in this column
//         // touches the pre-cluster in the neighbour column
//         if (neigh->area[jcolumn - 1].getYmin() > pcB->area[jcolumn].getYmax())
//           continue;
//         if (neigh->area[jcolumn - 1].getYmax() < pcB->area[jcolumn].getYmin())
//           continue;
//       }
//       pcB->paired = 2;
//       LOG(DEBUG) << "  adding column " << jcolumn << "  ib " << ib << "  to " << currentList;
//       neighList.push_back(pcB);
//       buildListOfNeighbours(jcolumn + 1, lastColumn, neighbours, skipPaired, currentList);
//       ++currentList;
//     }
//   }

//   return (neighbours.size() > 0);
// }

//______________________________________________________________________________
void Clusterizer::reset()
{
  /// Resets the clusters
  mNClusters = 0;
}
} // namespace mid
} // namespace o2
