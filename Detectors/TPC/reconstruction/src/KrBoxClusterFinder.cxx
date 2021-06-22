// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file KrBoxClusterFinder3D.cpp
/// \brief Class source code for Krypton and X-ray events
/// \author Philip Hauer <hauer@hiskp.uni-bonn.de>

#include "TPCReconstruction/KrBoxClusterFinder.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Utils.h"

#include "Framework/Logger.h"

#include <TFile.h>
#include <vector>

using namespace o2::tpc;

// If a gain map already exists in form of a CalDet file, it can be specified here
void KrBoxClusterFinder::loadGainMapFromFile(const std::string_view calDetFileName, const std::string_view gainMapName)
{
  auto calPads = utils::readCalPads(calDetFileName, gainMapName);
  auto gain = calPads[0];

  if (!gain) {
    LOGP(error, "No valid gain map object '{}' could be loaded from file '{}'", calDetFileName, gainMapName);
    return;
  }
  mGainMap.reset(gain);
  LOGP(info, "Loaded gain map object '{}' from file '{}'", calDetFileName, gainMapName);
}

void KrBoxClusterFinder::resetADCMap()
{
  // Reset whole map:
  for (int iTime = 0; iTime < MaxTimes; iTime++) {
    for (int iRow = 0; iRow < MaxRows; iRow++) {
      std::fill(mMapOfAllDigits[iTime][iRow].begin(), mMapOfAllDigits[iTime][iRow].end(), 0.);
      //for (int iPad = 0; iPad < MaxPads; iPad++) {
      //mMapOfAllDigits[iTime][iRow][iPad] = 0;
      //}
    }
  }
}

// Fill the map with all digits.
// You can pass a CalDet file to it so that the cluster finder can already correct for gain inhomogeneities
// The CalDet File should contain the relative Gain of each Pad.
void KrBoxClusterFinder::fillAndCorrectMap(std::vector<o2::tpc::Digit>& eventSector, const int sector)
{
  mSector = sector;
  resetADCMap();

  if (eventSector.size() == 0) {
    // prevents a segementation fault if the envent contains no data
    // segementation fault would occure later when trying to dereference the
    // max/min pointer (which are nullptr if data is empty) empty events should
    // be catched in the main function, hence, this "if" is no longer necessary
    LOGP(warning, "Sector size (amount of data points) in current run is 0!");
    LOGP(warning, "mMapOfAllDigits with 0's is generated in order to prevent a segementation fault.");

    return;
  }

  // Fill digits map
  for (const auto& digit : eventSector) {
    const int cru = digit.getCRU();
    const int time = digit.getTimeStamp();
    const int row = digit.getRow();
    const int pad = digit.getPad();
    const float adcValue = digit.getChargeFloat();

    fillADCValue(cru, row, pad, time, adcValue);
  }
}

void KrBoxClusterFinder::fillADCValue(int cru, int rowInSector, int padInRow, int timeBin, float adcValue)
{
  if (timeBin >= MaxTimes) {
    return;
  }

  // Every row starts at pad zero. But the number of pads in a row is not a constant.
  // If we would just fill the map naively, we would put pads next to each other, which are not neighbours on the pad plane.
  // Hence, we need to correct for this:
  mSector = cru / CRU::CRUperSector;
  const int pads = mMapperInstance.getNumberOfPadsInRowSector(rowInSector);
  const int corPad = padInRow - (pads / 2) + (MaxPads / 2);

  const auto correctionFactorCalDet = mGainMap.get();
  if (!correctionFactorCalDet) {
    mMapOfAllDigits[timeBin][rowInSector][corPad] = adcValue;
    return;
  }

  int padNum = mMapperInstance.globalPadNumber(PadPos(rowInSector, padInRow));
  float correctionFactor = correctionFactorCalDet->getValue(mSector, padNum);

  if (correctionFactor == 0) {
    LOGP(warning, "Encountered correction factor which is zero.");
    LOGP(warning, "Digit will be set to 0!");
    adcValue = 0;
  } else {
    adcValue /= correctionFactor;
  }

  mMapOfAllDigits[timeBin][rowInSector][corPad] = adcValue;
}

void KrBoxClusterFinder::init()
{
  const auto& param = KrBoxClusterFinderParam::Instance();

  mMaxClusterSizeTime = param.MaxClusterSizeTime;

  mMaxClusterSizeRowIROC = param.MaxClusterSizeRowIROC;
  mMaxClusterSizeRowOROC1 = param.MaxClusterSizeRowOROC1;
  mMaxClusterSizeRowOROC2 = param.MaxClusterSizeRowOROC2;
  mMaxClusterSizeRowOROC3 = param.MaxClusterSizeRowOROC3;

  mMaxClusterSizePadIROC = param.MaxClusterSizePadIROC;
  mMaxClusterSizePadOROC1 = param.MaxClusterSizePadOROC1;
  mMaxClusterSizePadOROC2 = param.MaxClusterSizePadOROC2;
  mMaxClusterSizePadOROC3 = param.MaxClusterSizePadOROC3;

  mQThresholdMax = param.QThresholdMax;
  mQThreshold = param.QThreshold;
  mMinNumberOfNeighbours = param.MinNumberOfNeighbours;
}

//#################################################

// Function to update the temporal cluster
void KrBoxClusterFinder::updateTempClusterFinal()
{
  if (mTempCluster.totCharge == 0) {
    mTempCluster.reset();
  } else {
    const float oneOverQtot = 1. / mTempCluster.totCharge;
    mTempCluster.meanPad *= oneOverQtot;
    mTempCluster.sigmaPad *= oneOverQtot;
    mTempCluster.meanRow *= oneOverQtot;
    mTempCluster.sigmaRow *= oneOverQtot;
    mTempCluster.meanTime *= oneOverQtot;
    mTempCluster.sigmaTime *= oneOverQtot;
    mTempCluster.sigmaPad = std::sqrt(std::abs(mTempCluster.sigmaPad - mTempCluster.meanPad * mTempCluster.meanPad));
    mTempCluster.sigmaRow = std::sqrt(std::abs(mTempCluster.sigmaRow - mTempCluster.meanRow * mTempCluster.meanRow));
    mTempCluster.sigmaTime = std::sqrt(std::abs(mTempCluster.sigmaTime - mTempCluster.meanTime * mTempCluster.meanTime));

    const int corPadsMean = mMapperInstance.getNumberOfPadsInRowSector(int(mTempCluster.meanRow));
    const int corPadsMaxCharge = mMapperInstance.getNumberOfPadsInRowSector(int(mTempCluster.maxChargeRow));

    // Since every padrow is shifted such that neighbouring pads are indeed neighbours, we have to shift once back:
    mTempCluster.meanPad = mTempCluster.meanPad + (corPadsMean / 2.0) - (MaxPads / 2.0);
    mTempCluster.maxChargePad = mTempCluster.maxChargePad + (corPadsMaxCharge / 2.0) - (MaxPads / 2.0);
    mTempCluster.sector = (decltype(mTempCluster.sector))mSector;
  }
}

// Function to update the temporal cluster.
void KrBoxClusterFinder::updateTempCluster(float tempCharge, int tempPad, int tempRow, int tempTime)
{
  if (tempCharge < mQThreshold) {
    LOGP(warning, "Update cluster was called but current charge is below mQThreshold");
    return;
  }

  // Some extrem ugly shaped clusters (mostly noise) might lead to an overflow.
  // Hence, we have to define an upper limit here:
  if (mTempCluster.size < 255) {
    mTempCluster.size += 1;
  }

  mTempCluster.totCharge += tempCharge;

  mTempCluster.meanPad += tempPad * tempCharge;
  mTempCluster.sigmaPad += tempPad * tempPad * tempCharge;

  mTempCluster.meanRow += tempRow * tempCharge;
  mTempCluster.sigmaRow += tempRow * tempRow * tempCharge;

  mTempCluster.meanTime += tempTime * tempCharge;
  mTempCluster.sigmaTime += tempTime * tempTime * tempCharge;

  if (tempCharge > mTempCluster.maxCharge) {
    mTempCluster.maxCharge = tempCharge;
    mTempCluster.maxChargePad = tempPad;
    mTempCluster.maxChargeRow = tempRow;
  }
}

// This function finds and evaluates all clusters in a 3D mMapOfAllDigits generated by the
// mMapOfAllDigitsCreator function, this function also updates the cluster tree
std::vector<std::tuple<int, int, int>> KrBoxClusterFinder::findLocalMaxima(bool directFilling)
{
  std::vector<std::tuple<int, int, int>> localMaximaCoords;
  // loop over whole mMapOfAllDigits the find clusters
  for (int iTime = 0; iTime < MaxTimes; iTime++) { //mMapOfAllDigits.size()
    const auto& mapRow = mMapOfAllDigits[iTime];
    for (int iRow = 0; iRow < MaxRows; iRow++) { // mapRow.size()
      // Since pad size is different for each ROC, we take this into account while looking for maxima:
      // setMaxClusterSize(iRow);
      if (iRow == 0) {
        mMaxClusterSizePad = mMaxClusterSizePadIROC;
        mMaxClusterSizeRow = mMaxClusterSizeRowIROC;
      } else if (iRow == MaxRowsIROC) {
        mMaxClusterSizePad = mMaxClusterSizePadOROC1;
        mMaxClusterSizeRow = mMaxClusterSizeRowOROC1;
      } else if (iRow == MaxRowsIROC + MaxRowsOROC1) {
        mMaxClusterSizePad = mMaxClusterSizePadOROC2;
        mMaxClusterSizeRow = mMaxClusterSizeRowOROC2;
      } else if (iRow == MaxRowsIROC + MaxRowsOROC1 + MaxRowsOROC2) {
        mMaxClusterSizePad = mMaxClusterSizePadOROC3;
        mMaxClusterSizeRow = mMaxClusterSizeRowOROC3;
      }

      const auto& mapPad = mapRow[iRow];
      const int padsInRow = mMapperInstance.getNumberOfPadsInRowSector(iRow);

      // Only loop over existing pads:
      for (int iPad = MaxPads / 2 - padsInRow / 2; iPad < MaxPads / 2 + padsInRow / 2; iPad++) { // mapPad.size()

        const float qMax = mapPad[iPad];

        // cluster Maximum must at least be larger than Threshold
        if (qMax <= mQThresholdMax) {
          continue;
        }

        // Acceptance condition: Require at least mMinNumberOfNeighbours neigbours
        // with signal in any direction!
        int noNeighbours = 0;
        if ((iPad + 1 < MaxPads) && (mMapOfAllDigits[iTime][iRow][iPad + 1] > mQThreshold)) {
          if (mMapOfAllDigits[iTime][iRow][iPad + 1] > qMax) {
            continue;
          }
          noNeighbours++;
        }
        if ((iPad - 1 >= 0) && (mMapOfAllDigits[iTime][iRow][iPad - 1] > mQThreshold)) {
          if (mMapOfAllDigits[iTime][iRow][iPad - 1] > qMax) {
            continue;
          }
          noNeighbours++;
        }
        if ((iRow + 1 < MaxRows) && (mMapOfAllDigits[iTime][iRow + 1][iPad] > mQThreshold)) {
          if (mMapOfAllDigits[iTime][iRow + 1][iPad] > qMax) {
            continue;
          }
          noNeighbours++;
        }
        if ((iRow - 1 >= 0) && (mMapOfAllDigits[iTime][iRow - 1][iPad] > mQThreshold)) {
          if (mMapOfAllDigits[iTime][iRow - 1][iPad] > qMax) {
            continue;
          }
          noNeighbours++;
        }
        if ((iTime + 1 < MaxTimes) && (mMapOfAllDigits[iTime + 1][iRow][iPad] > mQThreshold)) {
          if (mMapOfAllDigits[iTime + 1][iRow][iPad] > qMax) {
            continue;
          }
          noNeighbours++;
        }
        if ((iTime - 1 >= 0) && (mMapOfAllDigits[iTime - 1][iRow][iPad] > mQThreshold)) {
          if (mMapOfAllDigits[iTime - 1][iRow][iPad] > qMax) {
            continue;
          }
          noNeighbours++;
        }
        if (noNeighbours < mMinNumberOfNeighbours) {
          continue;
        }

        // Check that this is a local maximum
        // Note that the checking is done so that if 2 charges have the same
        // qMax then only 1 cluster is generated
        // (that is why there is BOTH > and >=)
        // -> only the maximum with the smalest indices will be accepted
        bool thisIsMax = true;

        for (int j = -mMaxClusterSizeTime; (j <= mMaxClusterSizeTime) && thisIsMax; j++) {
          if ((iTime + j >= MaxTimes) || (iTime + j < 0)) {
            continue;
          }
          for (int k = -mMaxClusterSizeRow; (k <= mMaxClusterSizeRow) && thisIsMax; k++) {
            if ((iRow + k >= MaxRows) || (iRow + k < 0)) {
              continue;
            }
            for (int i = -mMaxClusterSizePad; (i <= mMaxClusterSizePad) && thisIsMax; i++) {
              if ((iPad + i >= MaxPads) || (iPad + i < 0)) {
                continue;
              }
              if (mMapOfAllDigits[iTime + j][iRow + k][iPad + i] > qMax) {
                thisIsMax = false;
              }
            }
          }
        }

        if (!thisIsMax) {
          continue;
        } else {
          if (directFilling) {
            buildCluster(iPad, iRow, iTime, directFilling);
          } else {
            localMaximaCoords.emplace_back(std::make_tuple(iPad, iRow, iTime));
          }

          // If we have found a local maximum, we can also skip the next few entries:
          iPad += mMaxClusterSizePad;
        }
      }
    }
  }
  return localMaximaCoords;
}

// Calculate the total charge as the sum over the region:
//
//    o o o o o
//    o i i i o
//    o i C i o
//    o i i i o
//    o o o o o
//
// with qmax at the center C.
//
// The inner charge (i) we always add, but we only add the outer
// charge (o) if the neighboring inner bin (i) has a signal.
//

// for loop over whole cluster, to determine if a charge should be added
// conditions are extrapolation of the 5x5 cluster case to arbitrary
// cluster sizes in 3 dimensions
KrCluster KrBoxClusterFinder::buildCluster(int clusterCenterPad, int clusterCenterRow, int clusterCenterTime, bool directFilling)
{
  mTempCluster.reset();

  setMaxClusterSize(clusterCenterRow);

  // Loop over all neighbouring time bins:
  for (int iTime = -mMaxClusterSizeTime; iTime <= mMaxClusterSizeTime; iTime++) {
    // If we would look out of range, we skip
    if (clusterCenterTime + iTime < 0) {
      continue;
    } else if (clusterCenterTime + iTime >= MaxTimes) {
      break;
    }

    // Loop over all neighbouring row bins:
    for (int iRow = -mMaxClusterSizeRow; iRow <= mMaxClusterSizeRow; iRow++) {
      // First: Check again if we look over array boundaries:
      if (clusterCenterRow + iRow < 0) {
        continue;
      } else if (clusterCenterRow + iRow >= MaxRows) {
        break;
      }
      // Second: Check if we might look over ROC boundaries:
      else if (clusterCenterRow < MaxRowsIROC) {
        if (clusterCenterRow + iRow > MaxRowsIROC) {
          break;
        }
      } else if (clusterCenterRow < MaxRowsIROC + MaxRowsOROC1) {
        if (clusterCenterRow + iRow < MaxRowsIROC || clusterCenterRow + iRow >= MaxRowsIROC + MaxRowsOROC1) {
          continue;
        }
      } else if (clusterCenterRow < MaxRowsIROC + MaxRowsOROC1 + MaxRowsOROC2) {
        if (clusterCenterRow + iRow < MaxRowsIROC + MaxRowsOROC1 || clusterCenterRow + iRow >= MaxRowsIROC + MaxRowsOROC1 + MaxRowsOROC2) {
          continue;
        }
      } else if (clusterCenterRow < MaxRowsIROC + MaxRowsOROC1 + MaxRowsOROC2 + MaxRowsOROC3) {
        if (clusterCenterRow + iRow < MaxRowsIROC + MaxRowsOROC1 + MaxRowsOROC2) {
          continue;
        }
      }

      // Loop over all neighbouring pad bins:
      for (int iPad = -mMaxClusterSizePad; iPad <= mMaxClusterSizePad; iPad++) {
        // First: Check again if we might look outside of map:
        if (clusterCenterPad + iPad < 0) {
          continue;
        } else if (clusterCenterPad + iPad >= MaxPads) {
          break;
        }

        // Second: Check if charge is above threshold
        // Might be not necessary since we deal with pedestal subtracted data
        if (mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad] <= mQThreshold) {
          continue;
        }
        // If not, there are several cases which were explained (for 2D) in the header of the code.
        // The first one is for the diagonal. So, the digit we are investigating here is on the diagonal:
        if (std::abs(iTime) == std::abs(iPad) && std::abs(iTime) == std::abs(iRow)) {
          // Now we check, if the next inner digit has a signal above threshold:
          if (mMapOfAllDigits[clusterCenterTime + iTime - signnum(iTime)][clusterCenterRow + iRow - signnum(iRow)][clusterCenterPad + iPad - signnum(iPad)] > mQThreshold) {
            // If yes, the cluster gets updated with the digit on the diagonal.
            updateTempCluster(mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        }
        // Basically, we go through every possible case in the next few if-else conditions:
        else if (std::abs(iTime) == std::abs(iPad)) {
          if (mMapOfAllDigits[clusterCenterTime + iTime - signnum(iTime)][clusterCenterRow + iRow][clusterCenterPad + iPad - signnum(iPad)] > mQThreshold) {
            updateTempCluster(mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iTime) == std::abs(iRow)) {
          if (mMapOfAllDigits[clusterCenterTime + iTime - signnum(iTime)][clusterCenterRow + iRow - signnum(iRow)][clusterCenterPad + iPad] > mQThreshold) {
            updateTempCluster(mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iPad) == std::abs(iRow)) {
          if (mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow - signnum(iRow)][clusterCenterPad + iPad - signnum(iPad)] > mQThreshold) {
            updateTempCluster(mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iTime) > std::abs(iPad) && std::abs(iTime) > std::abs(iRow)) {
          if (mMapOfAllDigits[clusterCenterTime + iTime - signnum(iTime)][clusterCenterRow + iRow][clusterCenterPad + iPad] > mQThreshold) {
            updateTempCluster(mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iTime) < std::abs(iPad) && std::abs(iPad) > std::abs(iRow)) {
          if (mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad - signnum(iPad)] > mQThreshold) {
            updateTempCluster(mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iTime) < std::abs(iRow) && std::abs(iPad) < std::abs(iRow)) {
          if (mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow - signnum(iRow)][clusterCenterPad + iPad] > mQThreshold) {
            updateTempCluster(mMapOfAllDigits[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        }
      }
    }
  }
  // At the end, out mTempCluster should contain all digits that were assigned to the cluster.
  // So before returning it, we update it one last time to calculate the correct means and sigmas.
  updateTempClusterFinal();

  if (directFilling) {
    mClusters.emplace_back(mTempCluster);
  }

  return mTempCluster;
}

// Check if we are in IROC, OROC1, OROC2 or OROC3 and adapt the box size (max cluster size) accordingly.
void KrBoxClusterFinder::setMaxClusterSize(int row)
{
  if (row < MaxRowsIROC) {
    mMaxClusterSizePad = mMaxClusterSizePadIROC;
    mMaxClusterSizeRow = mMaxClusterSizeRowIROC;
  } else if (row < MaxRowsIROC + MaxRowsOROC1) {
    mMaxClusterSizePad = mMaxClusterSizePadOROC1;
    mMaxClusterSizeRow = mMaxClusterSizeRowOROC1;
  } else if (row < MaxRowsIROC + MaxRowsOROC1 + MaxRowsOROC2) {
    mMaxClusterSizePad = mMaxClusterSizePadOROC2;
    mMaxClusterSizeRow = mMaxClusterSizeRowOROC2;
  } else if (row < MaxRowsIROC + MaxRowsOROC1 + MaxRowsOROC2 + MaxRowsOROC3) {
    mMaxClusterSizePad = mMaxClusterSizePadOROC3;
    mMaxClusterSizeRow = mMaxClusterSizeRowOROC3;
  }
}
