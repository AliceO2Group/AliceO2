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
  int detID = hit.GetDetectorID();

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
  uint16_t chipIndex = static_cast<uint16_t>(detID);

  if (detID > mGeometry->getSize() || mGeometry->getSize() < 1) {
    LOG(debug) << "Invalid detector ID: " << detID;
    return; // invalid detector ID
  }
  const auto& matrix = mGeometry->getMatrixL2G(hit.GetDetectorID());

  math_utils::Vector3D<float> xyzPositionStart(matrix ^ (hit.GetPosStart())); // start position in sensor frame
  // math_utils::Vector3D<float> xyzPositionEnd(matrix ^ (hit.GetPos()));      // end position in sensor frame

  int row = 0; // Will be determined from start hit position
  int col = 0; // Will be determined from start hit position

  if (!sSegmentation->localToDetector(xyzPositionStart.X(), xyzPositionStart.Z(), row, col, mGeometry->getIOTOFLayer(detID))) {
    LOG(debug) << "Hit position out of bounds for detector ID " << detID;
    return; // hit is outside the active area
  }

  // Create the digit with time information
  int digID = mDigits->size();
  mDigits->emplace_back(chipIndex, static_cast<uint16_t>(row), static_cast<uint16_t>(col), charge, smearedTime);

  LOG(debug) << "Created digit #" << digID << " chip=" << chipIndex
             << " charge=" << charge << " time=" << smearedTime << " ns";

  // Add MC truth label
  if (mMCLabels) {
    o2::MCCompLabel lbl(hit.GetTrackID(), evID, srcID, false);
    mMCLabels->addElement(digID, lbl);
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
  // Create ROF record for the current event
  if (mROFRecords && mDigits && !mDigits->empty()) {
    o2::itsmft::ROFRecord rof;
    rof.setFirstEntry(0);
    rof.setNEntries(mDigits->size());
    rof.setBCData(mEventTime);
    mROFRecords->push_back(rof);
    LOG(debug) << "Created ROF record with " << mDigits->size() << " digits";
  }
}

} // namespace o2::iotof
