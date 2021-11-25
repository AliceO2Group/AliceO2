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

void KrBoxClusterFinder::createInitialMap(const gsl::span<const Digit> eventSector)
{
  mSetOfTimeSlices.clear();
  mThresholdInfo.clear();

  for (int iTimeSlice = 0; iTimeSlice <= 2 * mMaxClusterSizeTime; ++iTimeSlice) {
    addTimeSlice(eventSector, iTimeSlice);
  }
}

void KrBoxClusterFinder::fillADCValueInLastSlice(int cru, int rowInSector, int padInRow, float adcValue)
{
  auto& timeSlice = mSetOfTimeSlices.back();
  auto& thresholdInfo = mThresholdInfo.back();

  // Correct for pad offset:
  const int padsInRow = mMapperInstance.getNumberOfPadsInRowSector(rowInSector);
  const int corPad = padInRow - (padsInRow / 2) + (MaxPads / 2);

  if (adcValue > mQThresholdMax) {
    thresholdInfo.digitAboveThreshold = true;
    thresholdInfo.rowAboveThreshold[rowInSector] = true;
  }

  // Get correction factor from gain map:
  const auto correctionFactorCalDet = mGainMap.get();
  if (!correctionFactorCalDet) {
    timeSlice[rowInSector][corPad] = adcValue;
    return;
  }

  int padNum = mMapperInstance.globalPadNumber(PadPos(rowInSector, padInRow));
  float correctionFactor = correctionFactorCalDet->getValue(mSector, padNum);

  if (correctionFactor <= 0) {
    LOGP(warning, "Encountered correction factor which is zero.");
    LOGP(warning, "Digit will be set to 0!");
    adcValue = 0;
  } else {
    adcValue /= correctionFactor;
  }

  timeSlice[rowInSector][corPad] = adcValue;
}

void KrBoxClusterFinder::addTimeSlice(const gsl::span<const Digit> eventSector, const int timeSlice)
{
  mSetOfTimeSlices.emplace_back();
  mThresholdInfo.emplace_back();

  for (; mFirstDigit < eventSector.size(); ++mFirstDigit) {
    const auto& digit = eventSector[mFirstDigit];
    const int time = digit.getTimeStamp();
    if (time != timeSlice) {
      return;
    }

    const int cru = digit.getCRU();
    mSector = cru / CRU::CRUperSector;

    const int rowInSector = digit.getRow();
    const int padInRow = digit.getPad();
    const float adcValue = digit.getChargeFloat();

    fillADCValueInLastSlice(cru, rowInSector, padInRow, adcValue);
  }
}

void KrBoxClusterFinder::loopOverSector(const gsl::span<const Digit> eventSector, const int sector)
{
  mFirstDigit = 0;
  mSector = sector;

  createInitialMap(eventSector);
  for (int iTimeSlice = mMaxClusterSizeTime; iTimeSlice < mMaxTimes - mMaxClusterSizeTime; ++iTimeSlice) {
    // only search for a local maximum if the central time slice has at least one ADC above the charge threshold
    if (mThresholdInfo[mMaxClusterSizeTime].digitAboveThreshold) {
      findLocalMaxima(true, iTimeSlice);
    }
    popFirstTimeSliceFromMap();
    addTimeSlice(eventSector, iTimeSlice + mMaxClusterSizeTime + 1);

    // don't spend unnecessary time looping till mMaxTimes if there is no more data
    if (mFirstDigit >= eventSector.size()) {
      break;
    }
  }
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

  mCutMinSigmaTime = param.CutMinSigmaTime;
  mCutMaxSigmaTime = param.CutMaxSigmaTime;
  mCutMinSigmaPad = param.CutMinSigmaPad;
  mCutMaxSigmaPad = param.CutMaxSigmaPad;
  mCutMinSigmaRow = param.CutMinSigmaRow;
  mCutMaxSigmaRow = param.CutMaxSigmaRow;
  mCutMaxQtot = param.CutMaxQtot;
  mCutQtot0 = param.CutQtot0;
  mCutQtotSizeSlope = param.CutQtotSizeSlope;
  mCutMaxSize = param.CutMaxSize;
  mApplyCuts = param.ApplyCuts;

  if (param.GainMapFile.size()) {
    LOGP(info, "loading gain map '{}' from file {}", param.GainMapName, param.GainMapFile);
    loadGainMapFromFile(param.GainMapFile, param.GainMapName);
  }
}

//#################################################

// Function to update the temporal cluster
void KrBoxClusterFinder::updateTempClusterFinal(const int timeOffset)
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

    mTempCluster.meanTime += timeOffset;
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

// This function finds and evaluates all clusters in a 3D mSetOfTimeSlices generated by the
// mSetOfTimeSlicesCreator function, this function also updates the cluster tree
std::vector<std::tuple<int, int, int>> KrBoxClusterFinder::findLocalMaxima(bool directFilling, const int timeOffset)
{
  std::vector<std::tuple<int, int, int>> localMaximaCoords;

  const int iTime = mMaxClusterSizeTime;
  const auto& mapRow = mSetOfTimeSlices[iTime];
  const auto& thresholdInfo = mThresholdInfo[iTime];

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
    // skip rows that don't have charges above the threshold
    if (!thresholdInfo.rowAboveThreshold[iRow]) {
      continue;
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
      if ((iPad + 1 < MaxPads) && (mapPad[iPad + 1] > mQThreshold)) {
        if (mapPad[iPad + 1] > qMax) {
          continue;
        }
        noNeighbours++;
      }

      if ((iPad - 1 >= 0) && (mapPad[iPad - 1] > mQThreshold)) {
        if (mapPad[iPad - 1] > qMax) {
          continue;
        }
        noNeighbours++;
      }

      if ((iRow + 1 < MaxRows) && (mSetOfTimeSlices[iTime][iRow + 1][iPad] > mQThreshold)) {
        if (mSetOfTimeSlices[iTime][iRow + 1][iPad] > qMax) {
          continue;
        }
        noNeighbours++;
      }

      if ((iRow - 1 >= 0) && (mSetOfTimeSlices[iTime][iRow - 1][iPad] > mQThreshold)) {
        if (mSetOfTimeSlices[iTime][iRow - 1][iPad] > qMax) {
          continue;
        }
        noNeighbours++;
      }

      if ((iTime + 1 < mMaxTimes) && (mSetOfTimeSlices[iTime + 1][iRow][iPad] > mQThreshold)) {
        if (mSetOfTimeSlices[iTime + 1][iRow][iPad] > qMax) {
          continue;
        }
        noNeighbours++;
      }

      if ((iTime - 1 >= 0) && (mSetOfTimeSlices[iTime - 1][iRow][iPad] > mQThreshold)) {
        if (mSetOfTimeSlices[iTime - 1][iRow][iPad] > qMax) {
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
        if ((iTime + j >= mMaxTimes) || (iTime + j < 0)) {
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
            if (mSetOfTimeSlices[iTime + j][iRow + k][iPad + i] > qMax) {
              thisIsMax = false;
            }
          }
        }
      }

      if (!thisIsMax) {
        continue;
      } else {
        if (directFilling) {

          buildCluster(iPad, iRow, iTime, directFilling, timeOffset);
        } else {
          localMaximaCoords.emplace_back(std::make_tuple(iPad, iRow, iTime));
        }

        // If we have found a local maximum, we can also skip the next few entries:
        iPad += mMaxClusterSizePad;
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
KrCluster KrBoxClusterFinder::buildCluster(int clusterCenterPad, int clusterCenterRow, int clusterCenterTime, bool directFilling, const int timeOffset)
{
  mTempCluster.reset();
  setMaxClusterSize(clusterCenterRow);

  // Loop over all neighbouring time bins:
  for (int iTime = -mMaxClusterSizeTime; iTime <= mMaxClusterSizeTime; iTime++) {
    // Loop over all neighbouring row bins:

    for (int iRow = -mMaxClusterSizeRow; iRow <= mMaxClusterSizeRow; iRow++) {
      // First: Check if we look over array boundaries:
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
        if (mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad] <= mQThreshold) {
          continue;
        }
        // If not, there are several cases which were explained (for 2D) in the header of the code.
        // The first one is for the diagonal. So, the digit we are investigating here is on the diagonal:
        if (std::abs(iTime) == std::abs(iPad) && std::abs(iTime) == std::abs(iRow)) {
          // Now we check, if the next inner digit has a signal above threshold:
          if (mSetOfTimeSlices[clusterCenterTime + iTime - signnum(iTime)][clusterCenterRow + iRow - signnum(iRow)][clusterCenterPad + iPad - signnum(iPad)] > mQThreshold) {
            // If yes, the cluster gets updated with the digit on the diagonal.
            updateTempCluster(mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        }
        // Basically, we go through every possible case in the next few if-else conditions:
        else if (std::abs(iTime) == std::abs(iPad)) {
          if (mSetOfTimeSlices[clusterCenterTime + iTime - signnum(iTime)][clusterCenterRow + iRow][clusterCenterPad + iPad - signnum(iPad)] > mQThreshold) {
            updateTempCluster(mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iTime) == std::abs(iRow)) {
          if (mSetOfTimeSlices[clusterCenterTime + iTime - signnum(iTime)][clusterCenterRow + iRow - signnum(iRow)][clusterCenterPad + iPad] > mQThreshold) {
            updateTempCluster(mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iPad) == std::abs(iRow)) {
          if (mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow - signnum(iRow)][clusterCenterPad + iPad - signnum(iPad)] > mQThreshold) {
            updateTempCluster(mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iTime) > std::abs(iPad) && std::abs(iTime) > std::abs(iRow)) {
          if (mSetOfTimeSlices[clusterCenterTime + iTime - signnum(iTime)][clusterCenterRow + iRow][clusterCenterPad + iPad] > mQThreshold) {
            updateTempCluster(mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iTime) < std::abs(iPad) && std::abs(iPad) > std::abs(iRow)) {
          if (mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad - signnum(iPad)] > mQThreshold) {
            updateTempCluster(mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        } else if (std::abs(iTime) < std::abs(iRow) && std::abs(iPad) < std::abs(iRow)) {
          if (mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow - signnum(iRow)][clusterCenterPad + iPad] > mQThreshold) {
            updateTempCluster(mSetOfTimeSlices[clusterCenterTime + iTime][clusterCenterRow + iRow][clusterCenterPad + iPad], clusterCenterPad + iPad, clusterCenterRow + iRow, clusterCenterTime + iTime);
          }
        }
      }
    }
  }
  // At the end, out mTempCluster should contain all digits that were assigned to the cluster.
  // So before returning it, we update it one last time to calculate the correct means and sigmas.

  updateTempClusterFinal(timeOffset);

  if (directFilling) {
    if (!mApplyCuts || acceptCluster(mTempCluster)) {
      mClusters.emplace_back(mTempCluster);
    }
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

bool KrBoxClusterFinder::acceptCluster(const KrCluster& cl)
{
  // Qtot cut
  if (cl.totCharge > mCutMaxQtot) {
    return false;
  }

  // sigma cuts
  if (cl.sigmaPad < mCutMinSigmaPad || cl.sigmaPad > mCutMaxSigmaPad ||
      cl.sigmaRow < mCutMinSigmaRow || cl.sigmaRow > mCutMaxSigmaRow ||
      cl.sigmaTime < mCutMinSigmaTime || cl.sigmaRow > mCutMaxSigmaTime) {
    return false;
  }

  // total charge vs size cut
  if (cl.totCharge > mCutQtot0 + mCutQtotSizeSlope * cl.size) {
    return false;
  }

  // maximal size
  if (cl.size > mCutMaxSize) {
    return false;
  }

  return true;
}
