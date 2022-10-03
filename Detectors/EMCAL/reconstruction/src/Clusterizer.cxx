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

/// \file Clusterizer.cxx
/// \brief Implementation of the EMCAL clusterizer
#include <cstring>
#include <gsl/span>
#include <fairlogger/Logger.h> // for LOG
#include "EMCALReconstruction/Clusterizer.h"

using namespace o2::emcal;

///
/// Constructor
//____________________________________________________________________________
template <class InputType>
Clusterizer<InputType>::Clusterizer(double timeCut, double timeMin, double timeMax, double gradientCut, bool doEnergyGradientCut, double thresholdSeedE, double thresholdCellE) : mSeedList(), mInputMap(), mCellMask(), mTimeCut(timeCut), mTimeMin(timeMin), mTimeMax(timeMax), mGradientCut(gradientCut), mDoEnergyGradientCut(doEnergyGradientCut), mThresholdSeedEnergy(thresholdSeedE), mThresholdCellEnergy(thresholdCellE)
{
}

///
/// Default constructor
//____________________________________________________________________________
template <class InputType>
Clusterizer<InputType>::Clusterizer() : mSeedList(), mInputMap(), mCellMask(), mTimeCut(0), mTimeMin(0), mTimeMax(0), mGradientCut(0), mDoEnergyGradientCut(false), mThresholdSeedEnergy(0), mThresholdCellEnergy(0)
{
}

///
/// Initialize class member vars if not done in constructor
//____________________________________________________________________________
template <class InputType>
void Clusterizer<InputType>::initialize(double timeCut, double timeMin, double timeMax, double gradientCut, bool doEnergyGradientCut, double thresholdSeedE, double thresholdCellE)
{
  mTimeCut = timeCut;
  mTimeMin = timeMin;
  mTimeMax = timeMax;
  mGradientCut = gradientCut;
  mDoEnergyGradientCut = doEnergyGradientCut;
  mThresholdSeedEnergy = thresholdSeedE;
  mThresholdCellEnergy = thresholdCellE;
}

///
/// Recursively search for neighbours (EMCAL)
//____________________________________________________________________________
template <class InputType>
void Clusterizer<InputType>::getClusterFromNeighbours(std::vector<InputwithIndex>& clusterInputs, int row, int column)
{
  // Recursion 0, add seed cell/digit to cluster
  if (!clusterInputs.size()) {
    clusterInputs.emplace_back(mInputMap[row][column]);
  }

  // Mark the current cell as clustered
  mCellMask[row][column] = kTRUE;

  // Now go recursively to the next 4 neighbours and add them to the cluster if they fulfill the conditions
  constexpr int rowDiffs[4] = {-1, 0, 0, 1};
  constexpr int colDiffs[4] = {0, -1, 1, 0};
  for (int dir = 0; dir < 4; dir++) {
    if ((row + rowDiffs[dir] < 0) || (row + rowDiffs[dir] >= NROWS)) {
      continue;
    }
    if ((column + colDiffs[dir] < 0) || (column + colDiffs[dir] >= NCOLS)) {
      continue;
    }

    if (mInputMap[row + rowDiffs[dir]][column + colDiffs[dir]].mInput) {
      if (!mCellMask[row + rowDiffs[dir]][column + colDiffs[dir]]) {
        if (mDoEnergyGradientCut && not(mInputMap[row + rowDiffs[dir]][column + colDiffs[dir]].mInput->getEnergy() > mInputMap[row][column].mInput->getEnergy() + mGradientCut)) {
          if (not(TMath::Abs(mInputMap[row + rowDiffs[dir]][column + colDiffs[dir]].mInput->getTimeStamp() - mInputMap[row][column].mInput->getTimeStamp()) > mTimeCut)) {
            getClusterFromNeighbours(clusterInputs, row + rowDiffs[dir], column + colDiffs[dir]);
            // Add the cell/digit to the current cluster -- if we end up here, the selected cluster fulfills the condition
            clusterInputs.emplace_back(mInputMap[row + rowDiffs[dir]][column + colDiffs[dir]]);
          }
        }
      }
    }
  }
}

///
/// Get row (phi) and column (eta) of a cell/digit, values corresponding to topology
///
//____________________________________________________________________________
template <class InputType>
void Clusterizer<InputType>::getTopologicalRowColumn(const InputType& input, int& row, int& column)
{
  // Get SM number and relative row/column for SM
  auto cellIndex = mEMCALGeometry->GetCellIndex(input.getTower());
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
  } else {
    column += nSupMod % 2 * (48 + 1);
  }
}

///
/// Return number of found clusters. Start clustering from highest energy cell.
//____________________________________________________________________________
template <class InputType>
void Clusterizer<InputType>::findClusters(const gsl::span<InputType const>& inputArray)
{
  clear();

  // Algorithm
  // - Fill cells/digits in 2D topological map
  // - Fill struct arrays (energy,x,y)  (to get mapping energy -> (x,y))
  // - Create 2D bitmap (cell/digit is already clustered or not)
  // - Sort struct arrays with descending energy
  //
  // - Loop over arrays:
  // --> Check 2D bitmap (don't use cell/digit which are already clustered)
  // --> Take valid cell/digit with highest energy as seed (they are already sorted)
  // --> Recursive to neighboughs and create cluster
  // --> Seed cell and all neighbours belonging to cluster will be put in 2D bitmap

  // Reset cell/digit maps and cell masks
  // Loop over one array dim, then reset each array
  for (auto iArr = 0; iArr < NROWS; iArr++) {
    mCellMask[iArr].fill(kFALSE);
    mInputMap[iArr].fill({nullptr, -1});
  }

  // Calibrate cells/digits and fill the maps/arrays
  int nCells = 0;
  double ehs = 0.0;
  //for (auto dig : inputArray) {
  for (int iIndex = 0; iIndex < inputArray.size(); iIndex++) {

    auto& dig = inputArray[iIndex];

    Float_t inputEnergy = dig.getEnergy();
    Float_t time = dig.getTimeStamp();

    if (inputEnergy < mThresholdCellEnergy || time > mTimeMax || time < mTimeMin) {
      continue;
    }
    if (!mEMCALGeometry->CheckAbsCellId(dig.getTower())) {
      continue;
    }
    ehs += inputEnergy;

    // Put cell/digit to 2D map
    int row = 0, column = 0;
    getTopologicalRowColumn(dig, row, column);
    // not referencing dig here to get proper reference and not local copy
    mInputMap[row][column].mInput = inputArray.data() + iIndex; //
    mInputMap[row][column].mIndex = iIndex;                     // mInputMap saves the position of cells/digits in the input array
    mSeedList[nCells].energy = inputEnergy;
    mSeedList[nCells].row = row;
    mSeedList[nCells].column = column;
    nCells++;
  }

  // Sort struct arrays with ascending energy
  std::sort(mSeedList.begin(), std::next(std::begin(mSeedList), nCells));

  // Take next valid cell/digit in calorimeter as seed (in descending energy order)
  for (int i = nCells - 1; i >= 0; i--) {
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
    std::vector<InputwithIndex> clusterInputs;
    getClusterFromNeighbours(clusterInputs, row, column);

    // Add cells/digits for current cluster to cell/digit index vector
    int inputIndexStart = mInputIndices.size();
    for (auto dig : clusterInputs) {
      mInputIndices.emplace_back(dig.mIndex);
    }
    int inputIndexSize = mInputIndices.size() - inputIndexStart;

    // Now form cluster object from cells/digits
    mFoundClusters.emplace_back(mInputMap[row][column].mInput->getTimeStamp(), inputIndexStart, inputIndexSize); // Cluster object initialized w/ time of seed cell, start + size of associated cells
  }
  LOG(debug) << mFoundClusters.size() << "clusters found from " << nCells << " cells/digits (total=" << inputArray.size() << ")-> ehs " << ehs << " (minE " << mThresholdCellEnergy << ")";
}

template class o2::emcal::Clusterizer<o2::emcal::Cell>;
template class o2::emcal::Clusterizer<o2::emcal::Digit>;
