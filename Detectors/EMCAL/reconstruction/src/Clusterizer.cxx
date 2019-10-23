// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterizer.cxx
/// \brief Implementation of the EMCAL clusterizer
#include <cstring>
#include "FairLogger.h" // for LOG
#include "EMCALReconstruction/Clusterizer.h"

using namespace o2::emcal;

///
/// Constructor
//____________________________________________________________________________
Clusterizer::Clusterizer(double timeCut, double timeMin, double timeMax, double gradientCut, bool doEnergyGradientCut, double thresholdSeedE, double thresholdCellE) : mSeedList(), mDigitMap(), mCellMask(), mTimeCut(timeCut), mTimeMin(timeMin), mTimeMax(timeMax), mGradientCut(gradientCut), mDoEnergyGradientCut(doEnergyGradientCut), mThresholdSeedEnergy(thresholdSeedE), mThresholdCellEnergy(thresholdCellE)
{
}

///
/// Recursively search for neighbours (EMCAL)
//____________________________________________________________________________
void Clusterizer::getClusterFromNeighbours(std::vector<Digit*>& clusterDigits, int row, int column)
{
  // Recursion 0, add seed digit to cluster
  if (!clusterDigits.size()) {
    clusterDigits.emplace_back(mDigitMap[row][column]);
  }

  // Mark the current cell as clustered
  mCellMask[row][column] = kTRUE;

  // Now go recursively to the next 4 neighbours and add them to the cluster if they fulfill the conditions
  constexpr int rowDiffs[4] = {-1,0,0,1};
  constexpr int colDiffs[4] = {0,-1,1,0};
  for (int dir = 0; dir < 4; dir++) {
    if ((row + rowDiffs[dir] < 0) || (row + rowDiffs[dir] >= NROWS)) {
      continue;
    }
    if ((column + colDiffs[dir] < 0) || (column + colDiffs[dir] >= NCOLS)) {
      continue;
    }

    if (mDigitMap[row + rowDiffs[dir]][column + colDiffs[dir]]) {
      if (!mCellMask[row + rowDiffs[dir]][column + colDiffs[dir]]) {
        if (mDoEnergyGradientCut && not(mDigitMap[row + rowDiffs[dir]][column + colDiffs[dir]]->getEnergy() > mDigitMap[row][column]->getEnergy() + mGradientCut)) {
          if (not(TMath::Abs(mDigitMap[row + rowDiffs[dir]][column + colDiffs[dir]]->getTimeStamp() - mDigitMap[row][column]->getTimeStamp()) > mTimeCut)) {
            getClusterFromNeighbours(clusterDigits, row + rowDiffs[dir], column + colDiffs[dir]);
            // Add the digit to the current cluster -- if we end up here, the selected cluster fulfills the condition
            clusterDigits.emplace_back(mDigitMap[row + rowDiffs[dir]][column + colDiffs[dir]]);
          }
        }
      }
    }
  }
}

///
/// Get row (phi) and column (eta) of a digit, values corresponding to topology
///
//____________________________________________________________________________
void Clusterizer::getTopologicalRowColumn(const Digit& digit, int& row, int& column)
{
  // Get SM number and relative row/column for SM
  auto cellIndex = mEMCALGeometry->GetCellIndex(digit.getTower());
  int nSupMod = std::get<0>(cellIndex);

  auto phiEtaIndex = mEMCALGeometry->GetCellPhiEtaIndexInSModule(nSupMod, std::get<1>(cellIndex), std::get<2>(cellIndex), std::get<3>(cellIndex));
  row = std::get<0>(phiEtaIndex);
  column = std::get<1>(phiEtaIndex);

  // Add shifts wrt. supermodule and type of calorimeter
  // NOTE:
  // * Rows (phi) are arranged that one space is left empty between supermodules in phi
  //   This is due to the physical gap that forbids clustering
  // * For DCAL, there is an additional empty column between two supermodules in eta
  //   Again, this is to account for the gap in DCAL

  row += nSupMod / 2 * (24 + 1);
  // In DCAL, leave a gap between two SMs with same phi
  if (!mEMCALGeometry->IsDCALSM(nSupMod)) { // EMCAL
    column += nSupMod % 2 * 48;
  }
  else {
    column += nSupMod % 2 * (48 + 1);
  }
}

///
/// Return number of found clusters. Start clustering from highest energy cell.
//____________________________________________________________________________
void Clusterizer::findClusters(const std::vector<Digit>& digitArray)
{
  mFoundClusters.clear();
  mDigitIndices.clear();

  // Algorithm
  // - Fill digits in 2D topological map
  // - Fill struct arrays (energy,x,y)  (to get mapping energy -> (x,y))
  // - Create 2D bitmap (digit is already clustered or not)
  // - Sort struct arrays with descending energy
  //
  // - Loop over arrays:
  // --> Check 2D bitmap (don't use digit which are already clustered)
  // --> Take valid digit with highest energy as seed (they are already sorted)
  // --> Recursive to neighboughs and create cluster
  // --> Seed cell and all neighbours belonging to cluster will be put in 2D bitmap

  // Reset digit maps and cell masks
  // Loop over one array dim, then reset each array
  for (auto iArr = 0; iArr < NROWS; iArr++) {
    mCellMask[iArr].fill(0);
    mDigitMap[iArr].fill(0);
  }

  // Calibrate digits and fill the maps/arrays
  int nCells = 0;
  double ehs = 0.0;
  for (auto dig : digitArray) {
    Float_t digitEnergy = dig.getEnergy();
    Float_t time = dig.getTimeStamp();

    if (digitEnergy < mThresholdCellEnergy || time > mTimeMax || time < mTimeMin) {
      continue;
    }
    if (!mEMCALGeometry->CheckAbsCellId(dig.getTower())) {
      continue;
    }
    ehs += digitEnergy;

    // Put digit to 2D map
    int row = 0, column = 0;
    getTopologicalRowColumn(dig, row, column);
    mDigitMap[row][column] = &dig; // mDigitMap saves pointers to digits, therefore use addr operator here
    mSeedList[nCells].energy = digitEnergy;
    mSeedList[nCells].row = row;
    mSeedList[nCells].column = column;
    nCells++;
  }

  // Sort struct arrays with ascending energy
  std::sort(mSeedList.begin(), std::next(std::begin(mSeedList), nCells));
  //std::sort(mSeedList, mSeedList+nCells);

  // Take next valid digit in calorimeter as seed (in descending energy order)
  for (int i = nCells; i--;) {
    int row = mSeedList[i].row, column = mSeedList[i].column;
    // Continue if the cell is already masked (i.e. was already clustered)
    if (mCellMask[row][column]) {
      continue;
    }
    // Continue if energy constraints are not fulfilled
    if (mSeedList[i].energy <= mThresholdSeedEnergy) {
      continue;
    }

    // Seed is found, form cluster recursively
    std::vector<Digit*> clusterDigits;
    getClusterFromNeighbours(clusterDigits, row, column);

    // Add digits for current cluster to digit index vector
    int digitIndexStart = mDigitIndices.size();
    for (auto dig : clusterDigits) {
      mDigitIndices.emplace_back(dig->getTower());
    }
    int digitIndexSize = mDigitIndices.size() - digitIndexStart;

    // Now form cluster object from digits
    mFoundClusters.emplace_back(mDigitMap[row][column]->getTimeStamp(), digitIndexStart, digitIndexSize); // Cluster object initialized w/ time of seed cell, start + size of associated cells
  }
  LOG(DEBUG) << mFoundClusters.size() << "clusters found from " << nCells << " digits (total=" << digitArray.size() << ")-> ehs " << ehs << " (minE " << mThresholdCellEnergy << ")";
}
