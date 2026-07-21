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

///
/// \file Digitizer.cxx
/// \brief Implementation of the ALICE3 TOF digitizer
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-03-17
///

#include "IOTOFSimulation/Digitizer.h"
#include "IOTOFSimulation/DPLDigitizerParam.h"
#include "DetectorsRaw/HBFUtils.h"

#include <TRandom.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <fairlogger/Logger.h>

namespace o2::iotof
{

o2::iotof::Segmentation* Digitizer::sSegmentation = nullptr;
//_______________________________________________________________________
void Digitizer::init()
{
  const int numberOfChips = mGeometry->getSize();
  mChips.resize(numberOfChips);
  for (int i = numberOfChips; i--;) {
    mChips[i].setChipIndex(i);
    /// Noise map to be implemented
    /// if (mNoiseMap) {
    ///   mChips[i].setNoiseMap(mNoiseMap);
    /// }

    /// Dead channel map to be implemented
    /// if (mDeadChanMap) {
    ///   mChips[i].disable(mDeadChanMap->isFullChipMasked(i));
    ///   mChips[i].setDeadChanMap(mDeadChanMap);
    /// }
  }

  LOG(info) << "Initializing IOTOF digitizer";
  LOG(info) << "  Time resolution: " << mTimeResolution * 1e3 << " ps";
  LOG(info) << "  Charge threshold: " << mChargeThreshold << " electrons";
  LOG(info) << "  Detection efficiency: " << mEfficiency * 100 << " %";
  LOG(info) << "  Continuous mode: " << (mContinuous ? "ON" : "OFF");
  sSegmentation = o2::iotof::Segmentation::Instance();
}

//_______________________________________________________________________
void Digitizer::process(const std::vector<o2::itsmft::Hit>* hits, int evID, int srcID)
{
  // Digitize hits from a single event
  LOG(debug) << "Digitizing IOTOF hits: " << hits->size() << " hits from event " << evID << " source " << srcID;

  if (!hits || hits->empty()) {
    return;
  }

  // Sort hits by detector ID for better cache locality
  std::vector<int> hitIdx(hits->size());
  std::iota(hitIdx.begin(), hitIdx.end(), 0);
  std::sort(hitIdx.begin(), hitIdx.end(),
            [hits](int lhs, int rhs) {
              return (*hits)[lhs].GetDetectorID() < (*hits)[rhs].GetDetectorID();
            });

  // Process each hit
  for (int i : hitIdx) {
    processHit((*hits)[i], evID, srcID);
  }

  // In triggered mode, flush output after each event
  if (!mContinuous) {
    LOG(debug) << "Inner flushing for non-continuous mode";
    fillOutputContainer();
  }
}

//_______________________________________________________________________
void Digitizer::processHit(const o2::itsmft::Hit& hit, int evID, int srcID)
{
  // Process a single hit and create a digit if it passes all cuts

  // Apply efficiency cut
  if (!isEfficient()) {
    LOG(debug) << "Hit rejected by efficiency cut";
    return;
  }

  // Get detector element ID
  const int chipID = hit.GetDetectorID();
  auto& chip = mChips[chipID];
  if (chip.isDisabled()) {
    LOG(debug) << "Hit rejected because chip " << chipID << " is disabled";
    return;
  }

  // Convert energy loss to charge (number of electrons)
  float energyLoss = hit.GetEnergyLoss(); // in GeV
  int charge = energyToCharge(energyLoss);
  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();
  int electronsPerStep = static_cast<int>(charge / digitizerParams.nSimSteps);

  // Apply charge threshold
  if (charge < mChargeThreshold) {
    LOG(debug) << "Hit rejected by charge threshold: " << charge << " < " << mChargeThreshold;
    return;
  }

  // Get hit time and apply smearing
  // Hit time is in seconds, convert to ns and add event time
  double hitTime = hit.GetTime() * sec2ns;      // convert to ns
  double eventTimeNS = mEventTime.getTimeNS();  // event time since orbit 0
  double absoluteTime = hitTime + eventTimeNS;  // absolute time
  double smearedTime = smearTime(absoluteTime); // apply detector resolution

  if (chipID < 0 || chipID >= mGeometry->getSize() || mGeometry->getSize() < 1) {
    LOG(debug) << "Invalid detector ID: " << chipID << ", geometry size: " << mGeometry->getSize();
    return; // invalid detector ID
  }

  // Create the digit with time information
  o2::MCCompLabel label(hit.GetTrackID(), evID, srcID, false);
  const int roFrameAbs = 0; // For now, we can set this to 0 or calculate based on time if needed
  const int nROF = 1;       // For now, we can assume the signal is contained in one ROF, this can be extended to multiple ROFs based on the time

  float** respMatrix = nullptr;
  int rowStart = 0, colStart = 0, rowSpan = 0, colSpan = 0;
  stepping(hit, respMatrix, rowStart, colStart, rowSpan, colSpan);

  for (int irow = rowSpan; irow--;) {
    uint16_t rowIS = irow + rowStart;
    for (int icol = colSpan; icol--;) {
      uint16_t colIS = icol + colStart;
      float nEleResp = respMatrix[irow][icol];
      if (!nEleResp) {
        continue;
      }
      const int nElectronsSampled = gRandom->Poisson(electronsPerStep * nEleResp);
      // Noise can be added here if needed

      registerDigits(chip, roFrameAbs, smearedTime, nROF,
                     static_cast<uint16_t>(rowIS), static_cast<uint16_t>(colIS), nElectronsSampled, label);
    }
  }

  for (int irow = 0; irow < rowSpan; ++irow) {
    delete[] respMatrix[irow];
  }
  delete[] respMatrix;
}

void Digitizer::stepping(const o2::itsmft::Hit& hit, float**& respMatrix, int& rowStart, int& colStart, int& rowSpan, int& colSpan)
{
  const auto& matrix = mGeometry->getMatrixL2G(hit.GetDetectorID());
  const int chipID = hit.GetDetectorID();
  const int subdetectorID = mGeometry->getIOTOFLayer(chipID);

  auto xyzPositionStart(matrix ^ (hit.GetPosStart())); // start position in sensor frame
  auto xyzPositionEnd(matrix ^ (hit.GetPos()));        // end position in sensor frame

  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();
  const auto stepVector = (xyzPositionEnd - xyzPositionStart) / digitizerParams.nSimSteps;
  xyzPositionStart = xyzPositionStart + stepVector * 0.5f; // center the start position in the middle of the step
  xyzPositionEnd = xyzPositionEnd - stepVector * 0.5f;     // center the end position in the middle of the step

  rowStart = -1;
  colStart = -1;
  int rowEnd = -1, colEnd = -1, nSkip = 0, nSteps = digitizerParams.nSimSteps;
  while (!sSegmentation->localToDetector(xyzPositionStart.X(), xyzPositionStart.Z(), rowStart, colStart, mGeometry->getIOTOFLayer(chipID))) {
    if (++nSkip > digitizerParams.nSimSteps) { // additional check to add: should we exclude something?
      LOG(debug) << "Hit position out of bounds for detector ID " << chipID;
      return; // hit is outside the active area
    }
    xyzPositionStart += stepVector;
  }

  while (!sSegmentation->localToDetector(xyzPositionEnd.X(), xyzPositionEnd.Z(), rowEnd, colEnd, mGeometry->getIOTOFLayer(chipID))) {
    if (++nSkip > digitizerParams.nSimSteps) { // additional check to add: should we exclude something?
      LOG(debug) << "Hit position out of bounds for detector ID " << chipID;
      return; // hit is outside the active area
    }
    xyzPositionEnd += stepVector;
  }

  if (rowStart > rowEnd) {
    std::swap(rowStart, rowEnd);
  }
  if (colStart > colEnd) {
    std::swap(colStart, colEnd);
  }

  // Expand the range to take into account the effects of charge sharing
  rowStart -= digitizerParams.responseMatrixSize / 2;
  rowEnd += digitizerParams.responseMatrixSize / 2;
  rowStart = std::max(rowStart, 0);
  colStart = std::max(colStart, 0);

  rowEnd = std::min(rowEnd, (subdetectorID == 0 ? sSegmentation->mITofSpecsConfig.NRows : sSegmentation->mOTofSpecsConfig.NRows) - 1);
  colEnd = std::min(colEnd, (subdetectorID == 0 ? sSegmentation->mITofSpecsConfig.NCols : sSegmentation->mOTofSpecsConfig.NCols) - 1);
  rowSpan = rowEnd - rowStart + 1;
  colSpan = colEnd - colStart + 1;

  respMatrix = new float*[rowSpan];
  for (int i = 0; i < rowSpan; ++i) {
    respMatrix[i] = new float[colSpan]();
  }

  int rowPrev = -1, colPrev = -1, row = 0, col = 0;
  if (!respMatrix || rowSpan <= 0 || colSpan <= 0) {
    return;
  }
  if (nSkip) {
    nSteps -= nSkip;
  }

  auto& currentPosLocal = xyzPositionStart;
  for (int iStep = nSteps; iStep--;) {
    sSegmentation->localToDetector(currentPosLocal.X(), currentPosLocal.Z(), row, col, subdetectorID);
    if (row != rowPrev || col != colPrev) {
      rowPrev = row;
      colPrev = col;
    }

    currentPosLocal += stepVector; // Move to the next step position

    for (int irow = digitizerParams.responseMatrixSize; irow--;) {
      int rowDest = row + irow - (digitizerParams.responseMatrixSize / 2) - rowStart; // destination row in the respMatrix
      if (rowDest < 0 || rowDest >= rowSpan) {
        continue;
      }
      for (int icol = digitizerParams.responseMatrixSize; icol--;) {
        int colDest = col + icol - (digitizerParams.responseMatrixSize / 2) - colStart; // destination column in the respMatrix
        if (colDest < 0 || colDest >= colSpan) {
          continue;
        }
        respMatrix[rowDest][colDest] += 1.;
      }
    }
  }
}

//_______________________________________________________________________
double Digitizer::smearTime(double time) const
{
  // Apply Gaussian smearing to simulate detector time resolution
  if (mTimeResolution > 0) {
    return time + gRandom->Gaus(0, mTimeResolution);
  }
  return time;
}

//_______________________________________________________________________
int Digitizer::energyToCharge(float energyLoss) const
{
  // Convert energy loss (GeV) to number of electrons
  // Typical value: 3.6 eV per electron-hole pair in silicon
  // energyLoss is in GeV, mEnergyToCharge is GeV per electron
  return static_cast<int>(energyLoss / mEnergyToCharge);
}

//_______________________________________________________________________
bool Digitizer::isEfficient() const
{
  // Apply efficiency cut using random number
  return gRandom->Uniform() < mEfficiency;
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer()
{
  LOG(info) << "Filling output container with digits from chips";
  LOG(debug) << "Number of chips: " << mChips.size();

  o2::itsmft::ROFRecord rof;
  rof.setFirstEntry(mDigits->size()); // index of the first digit

  const auto* extraLabelBuffer = mExtraLabelBuffer.empty() ? nullptr : mExtraLabelBuffer.front().get();
  for (auto& chip : mChips) {

    if (chip.isDisabled()) {
      continue;
    }

    /// chip.addNoise(...); // to be implemented

    if (chip.isEmpty()) {
      continue;
    }

    auto& chipDigits = chip.getDigits();
    for (const auto& [key, digit] : chipDigits) {

      if (digit.getCharge() < mChargeThreshold) {
        continue; // skip digits below threshold
      }

      int digitID = mDigits->size();
      mDigits->emplace_back(digit.getChipIndex(), digit.getRow(), digit.getColumn(), digit.getCharge(), digit.getTime());
      if (mMCLabels) {
        mMCLabels->addElement(digitID, digit.getLabel().mLabel);
      }
      auto labelRef = digit.getLabel();

      while (mMCLabels && extraLabelBuffer != nullptr && labelRef.mNext >= 0) {
        labelRef = (*extraLabelBuffer)[labelRef.mNext];
        mMCLabels->addElement(digitID, labelRef.mLabel);
      }
    }
    chipDigits.clear(); // clear chip digits after copying to output
  }

  rof.setNEntries(mDigits->size() - rof.getFirstEntry()); // number of digits
  rof.setBCData(mContinuous ? mROFRecordIR : mEventTime);
  mROFRecords->push_back(rof);
  LOG(debug) << "Created ROF record with " << mDigits->size() << " digits";

  // extraLabelBuffer.clear(); // clear buffer for extra labels
  // mExtraLabelBuffer.emplace_back(mExtraLabelBuffer.front().release()); // move current buffer to the end
  // mExtraLabelBuffer.pop_front();
}

void Digitizer::registerDigits(Chip& chip, uint32_t roFrame, double time, int nROF,
                               uint16_t row, uint16_t col, int nElectrons, o2::MCCompLabel& label)
{
  (void)nROF;

  auto key = o2::iotof::Digit::getOrderingKey(chip.getChipIndex(), row, col);
  o2::iotof::LabeledDigit* existingDigit = chip.findDigit(key);
  if (!existingDigit) {
    // No existing digit, create a new one
    chip.addDigit(row, col, nElectrons, time, label);
  } else {
    // Digit already exists, update charge and labels
    const int storedCharge = existingDigit->getCharge();
    existingDigit->setCharge(storedCharge + nElectrons);
    existingDigit->setTime(std::min(existingDigit->getTime(), time));
    if (existingDigit->getLabel().mLabel == label) {
      return; // don't store the same label twice
    }
    std::vector<o2::iotof::McLabelRef>* extra = getExtraLabelBuffer(roFrame);
    auto labelRef = existingDigit->getLabel();
    const auto next = static_cast<int>(extra->size());
    extra->emplace_back(label, labelRef.mNext);
    labelRef.mNext = next;
    existingDigit->setLabel(labelRef);
  }
}

} // namespace o2::iotof
