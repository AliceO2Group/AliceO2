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
  int chipID = hit.GetDetectorID();
  auto& chip = mChips[chipID];
  if (chip.isDisabled()) {
    LOG(debug) << "Hit rejected because chip " << chipID << " is disabled";
    return;
  }

  // Convert energy loss to charge (number of electrons)
  float energyLoss = hit.GetEnergyLoss(); // in GeV
  int charge = energyToCharge(energyLoss);

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

  // For now, use simple row/col mapping from detector ID
  // TODO: Implement proper segmentation when geometry is finalized
  uint16_t chipIndex = static_cast<uint16_t>(chipID);

  if (chipID > mGeometry->getSize() || mGeometry->getSize() < 1) {
    LOG(debug) << "Invalid detector ID: " << chipID << ", geometry size: " << mGeometry->getSize();
    return; // invalid detector ID
  }
  const auto& matrix = mGeometry->getMatrixL2G(hit.GetDetectorID());

  math_utils::Vector3D<float> xyzPositionStart(matrix ^ (hit.GetPosStart())); // start position in sensor frame
  // math_utils::Vector3D<float> xyzPositionEnd(matrix ^ (hit.GetPos()));      // end position in sensor frame

  int row = 0; // Will be determined from start hit position
  int col = 0; // Will be determined from start hit position

  if (!sSegmentation->localToDetector(xyzPositionStart.X(), xyzPositionStart.Z(), row, col, mGeometry->getIOTOFLayer(chipID))) {
    LOG(debug) << "Hit position out of bounds for detector ID " << chipID;
    return; // hit is outside the active area
  }

  // Create the digit with time information
  int digID = mDigits->size();
  o2::MCCompLabel label(hit.GetTrackID(), evID, srcID, false);
  const int roFrameAbs = 0;  // For now, we can set this to 0 or calculate based on time if needed
  const int timeInitROF = 0; // For now, we can set this to 0 or calculate based on time if needed
  const int nROF = 1;        // For now, we can assume the signal is contained in one ROF, this can be extended to multiple ROFs based on the time

  registerDigits(chip, roFrameAbs, timeInitROF, nROF, static_cast<uint16_t>(row), static_cast<uint16_t>(col), charge, label);
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

  auto& extraLabelBuffer = *(mExtraLabelBuffer.front().get()); // buffer for extra labels
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

      /// Charge threshold not implemented yet
      /// if (digit.getCharge() < mChargeThreshold) {
      ///   continue; // skip digits below threshold
      /// }

      int digitID = mDigits->size();
      mDigits->emplace_back(digit.getChipIndex(), digit.getRow(), digit.getColumn(), digit.getCharge(), digit.getTime());
      mMCLabels->addElement(digitID, digit.getLabel().mLabel);
      auto labelRef = digit.getLabel();

      while (labelRef.mNext >= 0) {
        labelRef = extraLabelBuffer[labelRef.mNext];
        mMCLabels->addElement(digitID, labelRef.mLabel);
      }
    }
    chipDigits.clear(); // clear chip digits after copying to output
  }

  rof.setNEntries(mDigits->size() - rof.getFirstEntry()); // number of digits
  rof.setBCData(mEventTime);
  mROFRecords->push_back(rof);
  LOG(debug) << "Created ROF record with " << mDigits->size() << " digits";

  // extraLabelBuffer.clear(); // clear buffer for extra labels
  // mExtraLabelBuffer.emplace_back(mExtraLabelBuffer.front().release()); // move current buffer to the end
  // mExtraLabelBuffer.pop_front();
}

void Digitizer::registerDigits(Chip& chip, uint32_t roFrame, float timeInitROF, int nROF,
                               uint16_t row, uint16_t col, int nElectrons, o2::MCCompLabel& label)
{

  auto key = o2::iotof::Digit::getOrderingKey(roFrame, row, col);
  o2::iotof::LabeledDigit* existingDigit = chip.findDigit(key);
  if (!existingDigit) {
    // No existing digit, create a new one
    chip.addDigit(row, col, nElectrons, timeInitROF); // Last one should really just be time
  } else {
    // Digit already exists, update charge and labels
    const int storedCharge = existingDigit->getCharge();
    existingDigit->setCharge(storedCharge + nElectrons);
    if (existingDigit->getLabel().mLabel == label) {
      return; // don't store the same label twice
    }
    std::vector<o2::iotof::McLabelRef>* extra = getExtraLabelBuffer(roFrame);
    extra->emplace_back(label);
  }
}

} // namespace o2::iotof
