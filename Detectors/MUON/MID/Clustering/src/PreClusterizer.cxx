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

/// \file   MID/Clustering/src/PreClusterizer.cxx
/// \brief  Implementation of the pre-cluster reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 July 2018
#include "MIDClustering/PreClusterizer.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
void PreClusterizer::process(gsl::span<const ColumnData> stripPatterns, bool accumulate)
{
  // Reset fired DEs and pre-cluster information
  mActiveDEs.clear();
  if (!accumulate) {
    mPreClusters.clear();
  }

  // Load the stripPatterns to get the fired strips
  if (loadPatterns(stripPatterns)) {
    // Loop only on fired detection elements
    for (auto& deId : mActiveDEs) {
      // reset the precluster
      PatternStruct& de = mMpDEs[deId];

      preClusterizeNBP(de);
      preClusterizeBP(de);

      de.firedColumns = 0; // Reset fired columns
    }
  }
}

void PreClusterizer::process(gsl::span<const ColumnData> stripPatterns, gsl::span<const ROFRecord> rofRecords)
{
  mPreClusters.clear();
  mROFRecords.clear();
  for (auto& rofRecord : rofRecords) {
    auto firstEntry = mPreClusters.size();
    process(stripPatterns.subspan(rofRecord.firstEntry, rofRecord.nEntries), true);
    auto nEntries = mPreClusters.size() - firstEntry;
    mROFRecords.emplace_back(rofRecord, firstEntry, nEntries);
  }
}

//______________________________________________________________________________
bool PreClusterizer::loadPatterns(gsl::span<const ColumnData>& stripPatterns)
{

  // Loop on stripPatterns
  for (auto& col : stripPatterns) {
    auto& de = mMpDEs[col.deId];
    de.deId = col.deId;
    mActiveDEs.emplace(col.deId);

    de.firedColumns |= (1 << col.columnId);
    de.columns[col.columnId] = col;
  }

  return (stripPatterns.size() > 0);
}

//______________________________________________________________________________
void PreClusterizer::preClusterizeNBP(PatternStruct& de)
{
  PreCluster* pc = nullptr;
  for (int icolumn = 0; icolumn < 7; ++icolumn) {
    if (de.columns[icolumn].getNonBendPattern() == 0) {
      continue;
    }
    int nStripsNBP = mMapping.getNStripsNBP(icolumn, de.deId);
    for (int istrip = 0; istrip < nStripsNBP; ++istrip) {
      if (de.columns[icolumn].isNBPStripFired(istrip)) {
        if (!pc) {
          mPreClusters.push_back({de.deId, 1, static_cast<uint8_t>(icolumn), static_cast<uint8_t>(icolumn), 0, 0, static_cast<uint8_t>(istrip), static_cast<uint8_t>(istrip)});
          pc = &mPreClusters.back();
        }
        pc->lastColumn = icolumn;
        pc->lastStrip = istrip;
      } else {
        pc = nullptr;
      }
    }
    de.columns[icolumn].setNonBendPattern(0); // Reset pattern
  }
}

//______________________________________________________________________________
void PreClusterizer::preClusterizeBP(PatternStruct& de)
{
  for (int icolumn = mMapping.getFirstColumn(de.deId); icolumn < 7; ++icolumn) {
    if ((de.firedColumns & (1 << icolumn)) == 0) {
      continue;
    }
    PreCluster* pc = nullptr;
    int firstLine = mMapping.getFirstBoardBP(icolumn, de.deId);
    int lastLine = mMapping.getLastBoardBP(icolumn, de.deId);
    for (int iline = firstLine; iline <= lastLine; ++iline) {
      if (de.columns[icolumn].getBendPattern(iline) == 0) {
        continue;
      }
      for (int istrip = 0; istrip < 16; ++istrip) {
        if (de.columns[icolumn].isBPStripFired(istrip, iline)) {
          if (!pc) {
            mPreClusters.push_back({de.deId, 0, static_cast<uint8_t>(icolumn), static_cast<uint8_t>(icolumn), static_cast<uint8_t>(iline), static_cast<uint8_t>(iline), static_cast<uint8_t>(istrip), static_cast<uint8_t>(istrip)});
            pc = &mPreClusters.back();
          }
          pc->lastLine = iline;
          pc->lastStrip = istrip;
        } else {
          pc = nullptr;
        }
      }
      de.columns[icolumn].setBendPattern(0, iline); // Reset pattern

    } // loop on lines
  }   // loop on columns
}

} // namespace mid
} // namespace o2
