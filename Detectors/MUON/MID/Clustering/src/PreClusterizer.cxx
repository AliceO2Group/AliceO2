// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Clustering/src/PreClusterizer.cxx
/// \brief  Implementation of the pre-cluster reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 July 2018
#include "MIDClustering/PreClusterizer.h"

#include <cassert>
#include <fairlogger/Logger.h>

namespace o2
{
namespace mid
{

//______________________________________________________________________________
PreClusterizer::PreClusterizer()
  : mMapping(), mMpDEs(), mPreClusters(), mActiveDEs()
{
  /// Default constructor
}

//______________________________________________________________________________
bool PreClusterizer::process(const std::vector<ColumnData>& stripPatterns)
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

      PreClusters& pcs = nextPreCluster();

      // reset the precluster
      pcs.reset();
      pcs.setDEId(deIndex);
      PatternStruct& de = mMpDEs[deIndex];

      preClusterizeNBP(de, pcs);
      preClusterizeBP(de, pcs);

      de.firedColumns = 0; // Reset fired columns
    }
  }

  return true;
}

//______________________________________________________________________________
bool PreClusterizer::init()
{
  /// Initializes the class

  // Initialize pre-clusters
  mPreClusters.reserve(72);
  return true;
}

//______________________________________________________________________________
bool PreClusterizer::loadPatterns(const std::vector<ColumnData>& stripPatterns)
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
PreClusters& PreClusterizer::nextPreCluster()
{
  /// Iterates on pre-clusters
  if (mNPreClusters >= static_cast<int>(mPreClusters.size())) {
    mPreClusters.emplace_back(PreClusters());
    mPreClusters.back().init();
  }
  return mPreClusters[mNPreClusters++];
}

//______________________________________________________________________________
void PreClusterizer::preClusterizeNBP(PatternStruct& de, PreClusters& pcs)
{
  /// PreClusterizes non-bending plane
  PreClusters::PreClusterNBP* pc = nullptr;
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
          pc = pcs.nextPreClusterNBP();
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
                   << pc->area[icolumn].getYmax() << ")";
      } else {
        pc = nullptr;
      }
    }
    de.columns[icolumn].setNonBendPattern(0); // Reset pattern
  }
}

//______________________________________________________________________________
void PreClusterizer::preClusterizeBP(PatternStruct& de, PreClusters& pcs)
{
  /// PreClusterizes bending plane
  double limit = 0;
  for (int icolumn = mMapping.getFirstColumn(de.deId); icolumn < 7; ++icolumn) {
    if ((de.firedColumns & (1 << icolumn)) == 0) {
      continue;
    }
    PreClusters::PreClusterBP* pc = nullptr;
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
            pc = pcs.nextPreClusterBP(icolumn);
            pc->paired = 0;
            pc->column = icolumn;
            pc->area = mMapping.stripByLocation(istrip, 0, iline, icolumn, de.deId);
            limit = pc->area.getYmax();
            LOG(DEBUG) << "New precluster BP: DE  " << de.deId << "  icolumn " << icolumn;
          }
          pc->area.setYmax(limit);
          LOG(DEBUG) << "  adding line " << iline << "  strip " << istrip << "  (" << pc->area.getXmin()
                     << ", " << pc->area.getXmax() << ") (" << pc->area.getYmin() << ", "
                     << pc->area.getYmax() << ")";
        } else {
          pc = nullptr;
        }
      }
      de.columns[icolumn].setBendPattern(0, iline); // Reset pattern
    }                                               // loop on lines
  }                                                 // loop on columns
}

// //______________________________________________________________________________
// bool PreClusterizer::buildListOfNeighbours(int icolumn, int lastColumn, std::vector<std::vector<std::pair<int, int>>>& neighbours,
//                                            bool skipPaired, int currentList)
// {
//   /// Build list of neighbours
//   LOG(DEBUG) << "Building list of neighbours in (" << icolumn << ", " << lastColumn << ")";
//   for (int jcolumn = icolumn; jcolumn <= lastColumn; ++jcolumn) {
//     for (int ib = 0; ib < mNPreClusters[jcolumn]; ++ib) {
//       PreCluster& pcB = mPreClusters[jcolumn][ib];
//       if (skipPaired && pcB.paired > 0) {
//         LOG(DEBUG) << "Column " << jcolumn << "  ib " << ib << "  is already paired => skipPaired";
//         continue;
//       }
//       if (currentList >= neighbours.size()) {
//         // We are starting a new series of neighbour
//         // Let's make sure the pre-cluster is not already part of another list
//         if (pcB.paired == 2) {
//           LOG(DEBUG) << "Column " << jcolumn << "  ib " << ib << "  is already in a list";
//           continue;
//         }
//         LOG(DEBUG) << "New list " << currentList;
//         neighbours.emplace_back(std::vector<std::pair<int, int>>());
//       }
//       std::vector<PreCluster*>& neighList = neighbours[currentList];
//       if (!neighList.empty()) {
//         auto pair = neighList.back();
//         auto& neigh = mPreClusters[pair.first][pair.second];
//         if (neigh.area[jcolumn - 1].getYmin() > pcB.area[jcolumn].getYmax())
//           continue;
//         if (neigh.area[jcolumn - 1].getYmax() < pcB.area[jcolumn].getYmin())
//           continue;
//       }
//       pcB.paired = 2;
//       LOG(DEBUG) << "  adding column " << jcolumn << "  ib " << ib << "  to " << currentList;
//       neighList.emplace_back(jcolumn, ib);
//       buildListOfNeighbours(jcolumn + 1, lastColumn, neighbours, skipPaired, currentList);
//       ++currentList;
//     }
//   }

//   return (neighbours.size() > 0);
// }

//______________________________________________________________________________
void PreClusterizer::reset()
{
  /// Resets fired DEs

  // No need to reset the strip patterns here:
  // it is done while the patterns are processed to spare a loop
  mActiveDEs.clear();
  mNPreClusters = 0;
}
} // namespace mid
} // namespace o2
