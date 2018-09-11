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

#include "FairLogger.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
Clusterizer::Clusterizer()
  : mMapping(), mMpDEs(), mNPreClusters(), mPreClusters(), mActiveDEs(), mClusters(), mNClusters(0)
{
  /// Default constructor
}

//______________________________________________________________________________
bool Clusterizer::process(const std::vector<ColumnData>& stripPatterns)
{
  /// Main function: runs on a data containing the strip patterns
  /// and builds the clusters
  /// @param stripPatterns Vector of strip patterns per column

  // Reset fired DEs and cluster information
  reset();

  // Load the stripPatterns to get the fired strips
  if (loadPatterns(stripPatterns)) {
    // Loop only on fired detection elements
    for (auto& pair : mActiveDEs) {
      // loop on active DEs
      int deIndex = pair.first;

      // reset number of preclusters
      mNPreClusters.fill(0);
      PatternStruct& de = mMpDEs[deIndex];

      preClusterizeNBP(de);
      preClusterizeBP(de);

      makeClusters(deIndex);
      de.firedColumns = 0; // Reset fired columns
    }
  }

  return true;
}

//______________________________________________________________________________
bool Clusterizer::init()
{
  /// Initializes the class

  // prepare storage of clusters and PreClusters
  for (int iPlane = 0; iPlane < 2; ++iPlane) {
    mPreClusters[iPlane].reserve(100);
    mClusters.reserve(100);
  }

  return true;
}

//______________________________________________________________________________
bool Clusterizer::loadPatterns(const std::vector<ColumnData>& stripPatterns)
{
  /// Fills the mpDE structure with fired pads

  // Loop on stripPatterns
  for (auto col : stripPatterns) {
    int deIndex = col.deId;
    assert(deIndex < 72);

    auto search = mMpDEs.find(deIndex);
    PatternStruct* de = nullptr;
    if (search == mMpDEs.end()) {
      de = &mMpDEs[deIndex];
      de->deId = col.deId;
    } else {
      de = &(search->second);
    }

    mActiveDEs[deIndex] = 1;

    de->firedColumns |= (1 << col.columnId);
    de->columns[col.columnId] = col;
  }

  return (stripPatterns.size() > 0);
}

//______________________________________________________________________________
Clusterizer::PreCluster* Clusterizer::nextPreCluster(int icolumn)
{
  /// Iterates on pre-clusters
  if (mNPreClusters[icolumn] >= static_cast<int>(mPreClusters[icolumn].size())) {
    mPreClusters[icolumn].emplace_back(PreCluster());
  }
  PreCluster* pc = &(mPreClusters[icolumn][mNPreClusters[icolumn]]);
  ++mNPreClusters[icolumn];
  return pc;
}

//______________________________________________________________________________
void Clusterizer::preClusterizeNBP(PatternStruct& de)
{
  /// PreClusterizes non-bending plane
  PreCluster* pc = nullptr;
  double limit = 0;
  for (int icolumn = 0; icolumn < 7; ++icolumn) {
    if (de.columns[icolumn].getNonBendPattern() == 0) {
      continue;
    }
    int nStripsNBP = mMapping.getNStripsNBP(icolumn, de.deId);
    for (int istrip = 0; istrip < nStripsNBP; ++istrip) {
      if (de.columns[icolumn].isNBPStripFired(istrip)) {
        if (pc) {
          if (istrip == 0) {
            // We're changing column
            // In principle we could simply add the pitch, but for the cut RPCs
            // the y dimension changes as well in column 0
            pc->area[icolumn] = mMapping.stripByLocation(istrip, 1, 0, icolumn, de.deId);
            limit = pc->area[icolumn].getXmax();
          } else {
            limit += mMapping.getStripSize(istrip, 1, icolumn, de.deId);
          }
        } else {
          pc = nextPreCluster(7);
          pc->paired = 0;
          pc->firstColumn = icolumn;
          pc->area[icolumn] = mMapping.stripByLocation(istrip, 1, 0, icolumn, de.deId);
          limit = pc->area[icolumn].getXmax();
          LOG(DEBUG) << "New precluster NBP: DE  " << de.deId;
        }
        pc->lastColumn = icolumn;
        pc->area[icolumn].setXmax(limit);
        LOG(DEBUG) << "  adding col " << icolumn << "  strip " << istrip << "  (" << pc->area[icolumn].getXmin() << ", "
                   << pc->area[icolumn].getXmax() << ") (" << pc->area[icolumn].getYmin() << ", "
                   << pc->area[icolumn].getYmax();
      } else {
        pc = nullptr;
      }
    }
    de.columns[icolumn].setNonBendPattern(0); // Reset pattern
  }
}

//______________________________________________________________________________
void Clusterizer::preClusterizeBP(PatternStruct& de)
{
  /// PreClusterizes bending plane
  PreCluster* pc = nullptr;
  double limit = 0;
  for (int icolumn = mMapping.getFirstColumn(de.deId); icolumn < 7; ++icolumn) {
    if ((de.firedColumns & (1 << icolumn)) == 0) {
      continue;
    }
    int firstLine = mMapping.getFirstBoardBP(icolumn, de.deId);
    int lastLine = mMapping.getLastBoardBP(icolumn, de.deId);
    for (int iline = firstLine; iline <= lastLine; ++iline) {
      if (de.columns[icolumn].getBendPattern(iline) == 0) {
        continue;
      }
      for (int istrip = 0; istrip < 16; ++istrip) {
        if (de.columns[icolumn].isBPStripFired(istrip, iline)) {
          if (pc) {
            limit += mMapping.getStripSize(istrip, 0, icolumn, de.deId);
          } else {
            pc = nextPreCluster(icolumn);
            pc->paired = 0;
            pc->firstColumn = icolumn;
            pc->lastColumn = icolumn;
            pc->area[icolumn] = mMapping.stripByLocation(istrip, 0, iline, icolumn, de.deId);
            limit = pc->area[icolumn].getYmax();
            LOG(DEBUG) << "New precluster BP: DE  " << de.deId << "  icolumn " << icolumn;
          }
          pc->area[icolumn].setYmax(limit);
          LOG(DEBUG) << "  adding line " << iline << "  strip " << istrip << "  (" << pc->area[icolumn].getXmin()
                     << ", " << pc->area[icolumn].getXmax() << ") (" << pc->area[icolumn].getYmin() << ", "
                     << pc->area[icolumn].getYmax() << ")";
        } else {
          pc = nullptr;
        }
      }
      de.columns[icolumn].setBendPattern(0, iline); // Reset pattern
    }                                               // loop on lines
  }                                                 // loop on columns
}

//______________________________________________________________________________
void Clusterizer::makeClusters(const int& deIndex)
{
  /// Makes the clusters and stores it
  LOG(DEBUG) << "Clusterizing " << deIndex;

  // loop over pre-clusters in the non-bending plane
  for (int inb = 0; inb < mNPreClusters[7]; ++inb) {
    PreCluster& pcNB = mPreClusters[7][inb];
    int icolumn = pcNB.firstColumn;
    if (icolumn == pcNB.lastColumn) {
      // This is the most simple and common case: the NBP pre-cluster is on
      // on single column. So it can be easily matched with the BP
      // since the corresponding contours are both rectangles
      for (int ib = 0; ib < mNPreClusters[icolumn]; ++ib) {
        PreCluster& pcB = mPreClusters[icolumn][ib];
        makeCluster(pcB, pcNB, deIndex);
      }
    } else {
      // The NBP pre-cluster spans different columns.
      // The BP contour is therefore a serie of neighbour rectangles
      std::vector<std::vector<PreCluster*>> pcBneighbours;
      LOG(DEBUG) << "Spanning non-bend: " << icolumn << " -> " << pcNB.lastColumn;
      buildListOfNeighbours(icolumn, pcNB.lastColumn, pcBneighbours);
      for (const auto pcBlist : pcBneighbours) {
        makeCluster(pcBlist, deIndex, &pcNB);
      }
    }

    if (pcNB.paired > 0) {
      continue;
    }
    // If it is not paired, it means that we have
    // a monocathodic cluster in the NBP
    makeCluster(pcNB, pcNB, deIndex);
  } // loop over pre-clusters in the NBP

  /// Search for monocathodic clusters in the BP
  std::vector<std::vector<PreCluster*>> pcBneighbours;
  buildListOfNeighbours(0, 6, pcBneighbours, true);
  for (const auto pcBlist : pcBneighbours) {
    makeCluster(pcBlist, deIndex);
  }
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
void Clusterizer::makeCluster(PreCluster& clBend, PreCluster& clNonBend, const int& deIndex)
{
  /// Makes the cluster from pre-clusters
  Cluster2D& cl = nextCluster();
  int icolumn = clNonBend.firstColumn;
  cl.deId = (uint8_t)deIndex;
  cl.xCoor = 0.5 * (clNonBend.area[icolumn].getXmax() + clNonBend.area[icolumn].getXmin());
  cl.yCoor = 0.5 * (clBend.area[icolumn].getYmax() + clBend.area[icolumn].getYmin());
  double deltaX = clNonBend.area[icolumn].getXmax() - clNonBend.area[icolumn].getXmin();
  double deltaY = clBend.area[icolumn].getYmax() - clBend.area[icolumn].getYmin();
  cl.sigmaX2 = deltaX * deltaX / 12;
  cl.sigmaY2 = deltaY * deltaY / 12;
  clBend.paired = 1;
  clNonBend.paired = 1;

  LOG(DEBUG) << "pos: (" << cl.xCoor << ", " << cl.yCoor << ") err2: (" << cl.sigmaX2 << ", " << cl.sigmaY2 << ")";
}

//______________________________________________________________________________
void Clusterizer::makeCluster(std::vector<PreCluster*> pcBlist, const int& deIndex, PreCluster* clNonBend)
{
  /// Makes the cluster from pre-clusters
  // This is the general case:
  // perform the full calculation assuming a uniform charge distribution

  if (pcBlist.size() == 1) {
    // We fall back to the simple case:
    PreCluster* pcBP = pcBlist[0];
    PreCluster* pcNBP = clNonBend ? clNonBend : pcBP;
    makeCluster(*pcBP, *pcNBP, deIndex);
    return;
  }

  Cluster2D& cl = nextCluster();
  cl.deId = (uint8_t)deIndex;

  double x2[2][2] = { { 0., 0. }, { 0., 0. } };
  double x3[2][2] = { { 0., 0. }, { 0., 0. } };
  double dim[2][2];
  double delta[2];
  double sumArea = 0.;

  for (auto* pcBP : pcBlist) {
    PreCluster* pcNBP = clNonBend ? clNonBend : pcBP;
    int icolumn = pcBP->firstColumn;
    dim[0][0] = pcNBP->area[icolumn].getXmin();
    dim[0][1] = pcNBP->area[icolumn].getXmax();
    dim[1][0] = pcBP->area[icolumn].getYmin();
    dim[1][1] = pcBP->area[icolumn].getYmax();
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
    pcNBP->paired = 1;
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

//______________________________________________________________________________
bool Clusterizer::buildListOfNeighbours(int icolumn, int lastColumn, std::vector<std::vector<PreCluster*>>& neighbours,
                                        bool skipPaired, int currentList)
{
  /// Build list of neighbours
  LOG(DEBUG) << "Building list of neighbours in (" << icolumn << ", " << lastColumn << ")";
  for (int jcolumn = icolumn; jcolumn <= lastColumn; ++jcolumn) {
    for (int ib = 0; ib < mNPreClusters[jcolumn]; ++ib) {
      PreCluster* pcB = &mPreClusters[jcolumn][ib];
      if (skipPaired && pcB->paired > 0) {
        LOG(DEBUG) << "Column " << jcolumn << "  ib " << ib << "  is already paired => skipPaired";
        continue;
      }
      if (currentList >= neighbours.size()) {
        // We are starting a new series of neighbour
        // Let's make sure the pre-cluster is not already part of another list
        if (pcB->paired == 2) {
          LOG(DEBUG) << "Column " << jcolumn << "  ib " << ib << "  is already in a list";
          continue;
        }
        LOG(DEBUG) << "New list " << currentList;
        neighbours.emplace_back(std::vector<PreCluster*>());
      }
      std::vector<PreCluster*>& neighList = neighbours[currentList];
      if (!neighList.empty()) {
        auto* neigh = neighList.back();
        if (neigh->area[jcolumn - 1].getYmin() > pcB->area[jcolumn].getYmax())
          continue;
        if (neigh->area[jcolumn - 1].getYmax() < pcB->area[jcolumn].getYmin())
          continue;
      }
      pcB->paired = 2;
      LOG(DEBUG) << "  adding column " << jcolumn << "  ib " << ib << "  to " << currentList;
      neighList.push_back(pcB);
      buildListOfNeighbours(jcolumn + 1, lastColumn, neighbours, skipPaired, currentList);
      ++currentList;
    }
  }

  return (neighbours.size() > 0);
}

//______________________________________________________________________________
void Clusterizer::reset()
{
  /// Resets fired DEs and clusters

  // No need to reset the strip patterns here:
  // it is done while the patterns are processed to spare a loop
  mActiveDEs.clear();
  mNClusters = 0;
}
} // namespace mid
} // namespace o2
