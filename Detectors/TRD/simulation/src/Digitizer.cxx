// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDSimulation/Digitizer.h"

using namespace o2::trd;

Digitizer::Digitizer()
{
  hitLoopBegin = 0;
}

Digitizer::~Digitizer()
{
}

void Digitizer::process(std::vector<o2::trd::HitType> const& hits, std::vector<o2::trd::Digit>& digits)
{
  // (WIP) Implementation for digitization

  // Check if Geometry and if CCDB are available as they will be requiered
  /*
   mGeom = new TRDGeometry();
   TRDCalibDB * calibration = new TRDCalibDB();
   const int nTimeBins = calibration->GetNumberOfTimeBinsDCS();
  */

  // Sort hits according to detector number
  std::vector<o2::trd::HitType> hitCont = hits;
  if (!SortHits(hitCont)) {
    LOG(FATAL) << "Hits sorting failed";
    return;
  }

  const int kNdet = 540; // Get this from TRD Geometry

  for (int det = 0; det < kNdet; ++det) {
    // Loop over all TRD detectors

    // Jump to the next detector if the detector is
    // switched off, not installed, etc
    /*
    if (calibration->IsChamberNoData(det)) {
      continue;
    }
    if (!mGeo->ChamberInGeometry(det)) {
      continue
    }
    */

    // Hits are sorted for each detector, but you dont know how many hits each detector got
    std::vector<o2::trd::HitType> hitContInDet;
    if (!GetHitContainer(det, hitCont, hitContInDet) ||
        !CheckHitContainer(det, hitContInDet)) {
      break;
    }
    int signals; // dummy variable for now
    ConvertHits(det, hitContInDet, signals);

    digits.emplace_back();
  } // end of loop over detectors
}

bool Digitizer::SortHits(std::vector<o2::trd::HitType>& hitCont)
{
  std::sort(hitCont.begin(), hitCont.end(), [](const auto& a, const auto& b) {
    return a.GetDetectorID() < b.GetDetectorID();
  });
  return true;
}

bool Digitizer::GetHitContainer(const int det, const std::vector<o2::trd::HitType>& hitCont, std::vector<o2::trd::HitType>& hitContInDet)
{
  for (int i = hitLoopBegin; i < hitCont.size(); ++i) {
    auto hit = hitCont.at(i);
    if (hit.GetDetectorID() != det) {
      hitLoopBegin = i;
      break;
    }
    hitContInDet.push_back(hit);
  }
  return true;
}

bool Digitizer::CheckHitContainer(const int det, const std::vector<o2::trd::HitType>& hitContInDet)
{
  for (const auto& hit : hitContInDet) {
    if (det != hit.GetDetectorID()) {
      std::cout << "Digitizer::CheckHitContainer() Something when wront at the TRD digitizer" << std::endl;
      std::cout << "Digitizer::CheckHitContainer() DET = " << det << " and GetDetectorID = " << hit.GetDetectorID() << std::endl;
      return false;
    }
  }
  return true;
}

bool Digitizer::ConvertHits(int det, const std::vector<o2::trd::HitType>& hits, int& signal)
{
  // Convert the hit container associated to a given detector to signals

  return true;
}
