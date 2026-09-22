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

#include <TCollection.h>
#include <TFile.h>
#include <TKey.h>
#include <TRandom.h>


#include <set>
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

  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();
  if (!digitizerParams.efficiencyFilePath.empty()) {
    loadEfficiencyMap(digitizerParams.efficiencyFilePath);
  }

  LOG(info) << "Initializing IOTOF digitizer";
  LOG(info) << "  Time resolution: " << digitizerParams.timeResolution * 1e3 << " ps";
  LOG(info) << "  Charge threshold: " << digitizerParams.chargeThreshold << " electrons";
  LOG(info) << "  Detection efficiency: " << digitizerParams.efficiency * 100 << " %";
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

  // Get detector element ID
  const int chipID = hit.GetDetectorID();
  if (chipID < 0 || chipID >= mGeometry->getSize() || mGeometry->getSize() < 1) {
    LOG(debug) << "Invalid detector ID: " << chipID << ", geometry size: " << mGeometry->getSize();
    return; // invalid detector ID
  }
  const int subdetectorID = mGeometry->getIOTOFLayer(chipID);

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
  if (charge < digitizerParams.chargeThreshold) {
    LOG(debug) << "Hit rejected by charge threshold: " << charge << " < " << digitizerParams.chargeThreshold;
    return;
  }

  // Get hit time and apply smearing
  // Hit time is in seconds, convert to ns and add event time
  double hitTime = hit.GetTime() * sec2ns;                // convert to ns
  double eventTimeInBC = mEventTime.getTimeOffsetWrtBC(); // event time wrt bc
  double hitTimeWrtBC = hitTime + eventTimeInBC;          // hit time wrt bc
  double smearedTime = smearTime(hitTimeWrtBC);

  // Create the digit with time information
  o2::MCCompLabel label(hit.GetTrackID(), evID, srcID, false);
  const int roFrameAbs = 0; // For now, we can set this to 0 or calculate based on time if needed
  const int nROF = 1;       // For now, we can assume the signal is contained in one ROF, this can be extended to multiple ROFs based on the time

  float** respMatrix = nullptr;
  float** avgHitLocalX = nullptr;
  float** avgHitLocalZ = nullptr;
  int rowStart = 0, colStart = 0, rowSpan = 0, colSpan = 0;
  stepping(hit, respMatrix, avgHitLocalX, avgHitLocalZ, rowStart, colStart, rowSpan, colSpan);

  float xPixelCenter = 0.0f, zPixelCenter = 0.0f;
  for (int irow = rowSpan; irow--;) {
    uint16_t rowIS = irow + rowStart;
    for (int icol = colSpan; icol--;) {
      uint16_t colIS = icol + colStart;
      float nEleResp = respMatrix[irow][icol];
      if (!nEleResp) {
        continue;
      }

      // Apply efficiency cut based on the hit segment mean position relative to the pixel center
      sSegmentation->detectorToLocal(rowIS, colIS, xPixelCenter, zPixelCenter, subdetectorID);
      if (!isEfficient(avgHitLocalX[irow][icol] - xPixelCenter, avgHitLocalZ[irow][icol] - zPixelCenter)) {
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
    delete[] avgHitLocalX[irow];
    delete[] avgHitLocalZ[irow];
  }
  delete[] respMatrix;
  delete[] avgHitLocalX;
  delete[] avgHitLocalZ;
}

void Digitizer::stepping(const o2::itsmft::Hit& hit, float**& respMatrix, float**& avgHitLocalX, float**& avgHitLocalZ, int& rowStart, int& colStart, int& rowSpan, int& colSpan)
{
  LOG(debug) << "Stepping through hit for detector ID: " << hit.GetDetectorID();
  const int chipID = hit.GetDetectorID();
  const auto& matrix = mGeometry->getMatrixL2G(chipID);
  const int subdetectorID = mGeometry->getIOTOFLayer(chipID);

  LOG(debug) << "Transforming hit positions to sensor frame";
  auto xyzPositionStart(matrix ^ (hit.GetPosStart())); // start position in sensor frame
  auto xyzPositionEnd(matrix ^ (hit.GetPos()));        // end position in sensor frame

  LOG(debug) << "Hit start position in sensor frame: (" << xyzPositionStart.X() << ", " << xyzPositionStart.Y() << ", " << xyzPositionStart.Z() << ")";
  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();
  const auto stepVector = (xyzPositionEnd - xyzPositionStart) / digitizerParams.nSimSteps;
  xyzPositionStart = xyzPositionStart + stepVector * 0.5f; // center the start position in the middle of the step
  xyzPositionEnd = xyzPositionEnd - stepVector * 0.5f;     // center the end position in the middle of the step

  LOG(debug) << "Stepping vector: (" << stepVector.X() << ", " << stepVector.Y() << ", " << stepVector.Z() << ")";
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
  LOG(debug) << "Hit start position in sensor frame after adjustment: (" << xyzPositionStart.X() << ", " << xyzPositionStart.Y() << ", " << xyzPositionStart.Z() << ")";

  while (!sSegmentation->localToDetector(xyzPositionEnd.X(), xyzPositionEnd.Z(), rowEnd, colEnd, mGeometry->getIOTOFLayer(chipID))) {
    if (++nSkip > digitizerParams.nSimSteps) { // additional check to add: should we exclude something?
      LOG(debug) << "Hit position out of bounds for detector ID " << chipID;
      return; // hit is outside the active area
    }
    xyzPositionEnd -= stepVector;
  }
  LOG(debug) << "Hit end position in sensor frame after adjustment: (" << xyzPositionEnd.X() << ", " << xyzPositionEnd.Y() << ", " << xyzPositionEnd.Z() << ")";

  LOG(debug) << "Starting stepping through the hit with " << nSteps << " steps";
  if (nSkip) {
    nSteps -= nSkip;
  }
  LOG(debug) << "Adjusted number of steps after skipping: " << nSteps;

  std::set<int> crossedRows, crossedCols;
  for (int iStep = nSteps; iStep--;) {
    auto pixelCurrentPosLocal = xyzPositionStart + stepVector * iStep;
    int row, col;
    if (sSegmentation->localToDetector(pixelCurrentPosLocal.X(), pixelCurrentPosLocal.Z(), row, col, subdetectorID)) {
      crossedRows.insert(row);
      crossedCols.insert(col);
    }
  }
  LOG(debug) << "Crossed rows: ";
  for (const auto& row : crossedRows) {
    LOG(debug) << row;
  }
  LOG(debug) << "Crossed cols: ";
  for (const auto& col : crossedCols) {
    LOG(debug) << col;
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
  LOG(debug) << "Row range: [" << rowStart << ", " << rowEnd << "], Col range: [" << colStart << ", " << colEnd << "]";

  const auto& specsConfig = ChipSpecificsParam::Instance();
  rowEnd = std::min(rowEnd, (specsConfig.NRows) - 1);
  colEnd = std::min(colEnd, (specsConfig.NCols) - 1);
  rowSpan = rowEnd - rowStart + 1;
  colSpan = colEnd - colStart + 1;
  if (rowSpan <= 0 || colSpan <= 0) {
    return;
  }
  LOG(debug) << "Final row range: [" << rowStart << ", " << rowEnd << "], Col range: [" << colStart << ", " << colEnd << "]";

  respMatrix = new float*[rowSpan];
  avgHitLocalX = new float*[rowSpan];
  avgHitLocalZ = new float*[rowSpan];
  for (int i = 0; i < rowSpan; ++i) {
    respMatrix[i] = new float[colSpan]();
    avgHitLocalX[i] = new float[colSpan]();
    avgHitLocalZ[i] = new float[colSpan]();
  }
  LOG(debug) << "Allocated response matrix and average hit position arrays with size (" << rowSpan << ", " << colSpan << ")";

  if (!respMatrix || !avgHitLocalX || !avgHitLocalZ) {
    return;
  }

  int rowPrev = -1, colPrev = -1, row = 0, col = 0, nSkipPassive = 0;
  auto pixelStartPosLocal = xyzPositionStart;
  auto pixelCurrentPosLocal = xyzPositionStart;
  for (int iStep{0}; iStep < nSteps; ++iStep) {
    pixelCurrentPosLocal = xyzPositionStart + iStep * stepVector;

    // Step does not contribute if it is in the passive area
    if (!sSegmentation->localToDetector(pixelCurrentPosLocal.X(), pixelCurrentPosLocal.Z(), row, col, subdetectorID)) {
      LOG(debug) << "Step is in passive area: (" << pixelCurrentPosLocal.X() << ", " << pixelCurrentPosLocal.Z() << ") is outside the active area of chip " << subdetectorID;
      nSkipPassive++;
      continue;
    }

    // The step has reached another pixel, compute mean hit segment positions
    // for pixel efficiency evaluation and reset the start position for the next pixel
    LOG(debug) << "iStep: " << iStep << ", Current pixel: (row,col) = (" << row << ", " << col << "), Previous pixel: (rowPrev,colPrev) = (" << rowPrev << ", " << colPrev << ")";
    if (row != rowPrev || col != colPrev) {

      // Finalize the previous pixel
      if (rowPrev != -1 && colPrev != -1) {
        const int irow = rowPrev - rowStart;
        const int icol = colPrev - colStart;
        avgHitLocalX[irow][icol] = 0.5f * (pixelStartPosLocal.X() + pixelCurrentPosLocal.X() - (nSkipPassive + 1) * stepVector.X());
        avgHitLocalZ[irow][icol] = 0.5f * (pixelStartPosLocal.Z() + pixelCurrentPosLocal.Z() - (nSkipPassive + 1) * stepVector.Z());
        LOG(debug) << "avgHitLocalX = " << avgHitLocalX[irow][icol] << ", avgHitLocalZ = " << avgHitLocalZ[irow][icol];
        pixelStartPosLocal = pixelCurrentPosLocal;
        nSkipPassive = 0;
      }

      // Start the new pixel
      rowPrev = row;
      colPrev = col;
    }

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
  LOG(debug) << "Finished stepping through the hit for detector ID: " << chipID;

  LOG(debug) << "rowPrev: " << rowPrev << ", colPrev: " << colPrev << ", rowStart: " << rowStart << ", colStart: " << colStart;
  // Finalize the last pixel
  if (rowPrev != -1 && colPrev != -1) {
    const int irow = rowPrev - rowStart;
    const int icol = colPrev - colStart;
    // Sizes of avgHitLocalX, avgHitLocalZ
    LOG(debug) << "avgHitLocalX dimensions: " << rowSpan << " x " << colSpan;
    LOG(debug) << "avgHitLocalZ dimensions: " << rowSpan << " x " << colSpan;
    LOG(debug) << "Finalizing last pixel at (row,col) = (" << rowPrev << ", " << colPrev << ") with indices (irow,icol) = (" << irow << ", " << icol << ")";
    avgHitLocalX[irow][icol] = 0.5f * (pixelStartPosLocal.X() + pixelCurrentPosLocal.X() - nSkipPassive * stepVector.X());
    avgHitLocalZ[irow][icol] = 0.5f * (pixelStartPosLocal.Z() + pixelCurrentPosLocal.Z() - nSkipPassive * stepVector.Z());
    LOG(debug) << "Finalized last pixel average positions: avgHitLocalX = " << avgHitLocalX[irow][icol] << ", avgHitLocalZ = " << avgHitLocalZ[irow][icol];
  }
  LOG(debug) << "Finalized last pixel for detector ID: " << chipID;
}

//_______________________________________________________________________
double Digitizer::smearTime(double time) const
{
  // Apply Gaussian smearing to simulate detector time resolution
  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();
  if (digitizerParams.timeResolution > 0) {
    return time + gRandom->Gaus(0, digitizerParams.timeResolution);
  }
  return time;
}

//_______________________________________________________________________
int Digitizer::energyToCharge(float energyLoss) const
{
  // Convert energy loss (GeV) to number of electrons
  // Typical value: 3.6 eV per electron-hole pair in silicon
  // energyLoss is in GeV, energyToNElectrons is electrons per GeV
  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();
  return static_cast<int>(energyLoss * digitizerParams.energyToNElectrons);
}

//_______________________________________________________________________
void Digitizer::loadEfficiencyMap(const std::string& filePath)
{
  // Load the efficiency map from a file
  TFile* file = TFile::Open(filePath.c_str());
  if (!file || !file->IsOpen()) {
    LOG(error) << "Failed to open efficiency map file: " << filePath;
    return;
  }

  auto* rawMap = dynamic_cast<TH2D*>(file->Get("hEfficiencyMap"));
  if (!rawMap) {
    LOG(error) << "Failed to retrieve efficiency map from file: " << filePath;
    LOG(error) << "Available keys in the file:";
    TIter next(file->GetListOfKeys());
    TKey* key;
    while ((key = dynamic_cast<TKey*>(next()))) {
      LOG(error) << "  " << key->GetName() << " (" << key->GetClassName() << ")";
    }
    file->Close();
    return;
  }
  mEfficiencyMap = dynamic_cast<TH2D*>(rawMap->Clone("mEfficiencyMap"));
  mEfficiencyMap->SetDirectory(nullptr); // Detach from file to avoid deletion when file is closed

  file->Close();
}

//_______________________________________________________________________
bool Digitizer::isEfficient(const float x, const float z) const
{
  // Apply efficiency cut using random number
  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();
  if (mEfficiencyMap) {
    // int bin = mEfficiencyMap->FindBin(x * o2::iotof::Digitizer::cm2um, z * o2::iotof::Digitizer::cm2um);
    int bin = mEfficiencyMap->FindBin(x * o2::iotof::Digitizer::cm2um, z * o2::iotof::Digitizer::cm2um);
    float efficiency = mEfficiencyMap->GetBinContent(bin);
    LOG(debug) << "Efficiency map check: x=" << x * o2::iotof::Digitizer::cm2um << ", z=" << z * o2::iotof::Digitizer::cm2um << ", bin=" << bin << ", efficiency=" << efficiency;
    return gRandom->Uniform() < efficiency;
  }
  return gRandom->Uniform() < digitizerParams.efficiency;
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer()
{
  LOG(info) << "Filling output container with digits from chips";
  LOG(debug) << "Number of chips: " << mChips.size();

  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();

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

      if (digit.getCharge() < digitizerParams.chargeThreshold) {
        continue; // skip digits below threshold
      }

      int digitID = mDigits->size();
      mDigits->emplace_back(digit.getChipIndex(), digit.getRow(), digit.getColumn(), digit.getCharge(), digit.getTime(), digit.getBc(), digit.getTdc());
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

  const auto& digitizerParams = o2::iotof::DPLDigitizerParam::Instance();

  uint64_t nbc = static_cast<uint64_t>(time / o2::constants::lhc::LHCBunchSpacingNS);
  int tdc = int((time - nbc * o2::constants::lhc::LHCBunchSpacingNS) / digitizerParams.tdcBin);
  nbc += mEventTime.toLong();

  LOG(debug) << "nbc: " << nbc << "\ttdc: " << tdc;
  double absoluteTime = tdc * digitizerParams.tdcBin * 1.e-9 + nbc * o2::constants::lhc::LHCBunchSpacingNS;

  auto key = o2::iotof::Digit::getOrderingKey(nbc, row, col);
  o2::iotof::LabeledDigit* existingDigit = chip.findDigit(key);
  if (!existingDigit) {
    // No existing digit, create a new one
    chip.addDigit(row, col, nElectrons, absoluteTime, nbc, tdc, label);
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
