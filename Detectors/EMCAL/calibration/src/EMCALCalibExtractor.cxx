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

#include "EMCALCalibration/EMCALCalibExtractor.h"

namespace o2
{
namespace emcal
{

//-------------------------------------------------------------------------------------------
// This function builds the scaled hit distribution
// It normalizes the hits/cell to the mean value of the row and the column of the cell
// this is done in an iterative procedure (about 5 iterations are needed)
// with this procedure, the effects of the TRD and SM structures on the EMCal can be minimized
// The output is a  boost histogram with the hits/cell as a function of cell ID.

/// \param emin -- min. energy for cell amplitudes
/// \param emax -- max. energy for cell amplitudes
// ------------------------------------------------------------------------------------------
boostHisto EMCALCalibExtractor::buildHitAndEnergyMeanScaled(double emin, double emax, boostHisto cellAmplitude)
{
  // create the output histogram
  boostHisto eSumHistoScaled = boost::histogram::make_histogram(boost::histogram::axis::regular<>(100, 0, 100, "t-texp"), boost::histogram::axis::integer<>(0, mNcells, "CELL ID"));
  // create a slice for each cell with energies ranging from emin to emax
  auto hEnergyCol = boost::histogram::make_histogram(boost::histogram::axis::regular<>(100, 0, 100., "t-texp"));
  auto hEnergyRow = boost::histogram::make_histogram(boost::histogram::axis::regular<>(250, 0, 250., "t-texp"));
  // temp histogram used to get the scaled energies
  auto hEnergyScaled = boost::histogram::make_histogram(boost::histogram::axis::regular<>(100, 0, 100, "t-texp"), boost::histogram::axis::integer<>(0, mNcells, "CELL ID"));

  //...........................................
  // start iterative process of scaling of cells
  //...........................................
  for (int iter = 1; iter < 5; iter++) {
    // array of vectors for calculating the mean hits per col/row
    std::vector<double> vecCol[100];
    std::vector<double> vecRow[250];

    for (int cellID = 0; cellID < mNcells; cellID++) {
      auto tempSlice = boost::histogram::algorithm::reduce(cellAmplitude, boost::histogram::algorithm::shrink(cellID, cellID), boost::histogram::algorithm::shrink(emin, emax));
      // (0 - row, 1 - column)
      auto position = mGeometry->GlobalRowColFromIndex(cellID);
      int row = std::get<0>(position);
      int col = std::get<1>(position);

      double dCellEnergy = 0.;
      double dNumOfHits = 0.;

      // will need to change this from 100 to a normal value for the energy axis
      auto energyAxis = boost::histogram::axis::regular<>(100, 0, 100, "t-texp");
      auto eMinIndex = energyAxis.index(emin);
      auto eMaxIndex = energyAxis.index(emax);
      for (int EBin = eMinIndex; EBin < eMaxIndex; EBin++) {
        dCellEnergy += hEnergyScaled.at(EBin, cellID) * energyAxis.value(EBin);
        dNumOfHits += hEnergyScaled.at(EBin, cellID);
      }

      if (dCellEnergy > 0.) {
        hEnergyCol(col + 0.5, dCellEnergy);
        hEnergyRow(row + 0.5, dCellEnergy);
        vecCol[col].push_back(dCellEnergy);
        vecRow[row].push_back(dCellEnergy);
      }
    } // end loop over cells

    // Fill the histogram: mean hit per column
    for (int iCol = 1; iCol <= 100; iCol++) {
      if (vecCol[iCol - 1].size() > 0.) {
        auto colAxis = boost::histogram::axis::regular<>(100, 0, 100., "t-texp");
        auto indexCol = colAxis.index(iCol - 0.5);
        hEnergyCol(indexCol, hEnergyCol.at(indexCol) / vecCol[iCol - 1].size());
      }
    }

    // Fill the histogram: mean hit per row
    for (int iRow = 1; iRow <= 250; iRow++) {
      if (vecRow[iRow - 1].size() > 0.) {
        auto rowAxis = boost::histogram::axis::regular<>(250, 0, 250., "t-texp");
        auto indexRow = rowAxis.index(iRow - 0.5);
        hEnergyRow(indexRow, hEnergyRow.at(indexRow) / vecRow[iRow - 1].size());
      }
    }

    // in run2 there was now a global fit to hits per row
    // could we just do the mean?
    auto rowResult = o2::utils::fitBoostHistoWithGaus<double>(hEnergyRow);
    double meanValRow = rowResult.at(1);

    auto colResult = o2::utils::fitBoostHistoWithGaus<double>(hEnergyCol);
    double meanValCol = colResult.at(1);

    // Scale each cell by the deviation of the mean of the column and the global mean
    for (int iCell = 0; iCell < mNcells; iCell++) {
      // (0 - row, 1 - column)
      auto position = mGeometry->GlobalRowColFromIndex(iCell);
      int col = std::get<1>(position);
      if (hEnergyCol.at(col) > 0.) {
        // will need to change the 100 depending on the number of energy bins we end up having
        for (int EBin = 1; EBin < 100; EBin++) {
          hEnergyScaled(EBin, iCell, hEnergyScaled.at(EBin, iCell) * (meanValCol / hEnergyCol.at(col)));
        }
      }
    }

    // Scale each cell by the deviation of the mean of the row and the global mean
    for (int iCell = 0; iCell < mNcells; iCell++) {
      // (0 - row, 1 - column)
      auto position = mGeometry->GlobalRowColFromIndex(iCell);
      int row = std::get<0>(position);
      if (hEnergyRow.at(row) > 0.) {
        // will need to change the 100 depending on the number of energy bins we end up having
        for (int EBin = 1; EBin < 100; EBin++) {
          hEnergyScaled(EBin, iCell, hEnergyScaled.at(EBin, iCell) * (meanValRow / hEnergyRow.at(row)));
        }
      }
    }

    //....................
    // iteration ends here
    //....................

  } // end loop iters

  //............................................................................................
  //..here the average hit per event and the average energy per hit is caluclated for each cell.
  //............................................................................................
  for (Int_t cell = 0; cell < mNcells; cell++) {
    Double_t Esum = 0;
    Double_t Nsum = 0;

    for (Int_t j = 1; j <= 100; j++) {
      auto energyAxis = boost::histogram::axis::regular<>(100, 0, 100, "t-texp");
      Double_t E = energyAxis.value(j);
      Double_t N = hEnergyScaled.at(j, cell);
      if (E < emin || E > emax) {
        continue;
      }
      Esum += E * N;
      Nsum += N;
    }
    if (Nsum > 0.) {
      eSumHistoScaled(cell, Esum / (Nsum)); //..average energy per hit
    }
  }

  return eSumHistoScaled;
}
//____________________________________________

} // end namespace emcal
} // end namespace o2
