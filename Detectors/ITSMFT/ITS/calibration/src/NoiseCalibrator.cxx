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

/// @file   NoiseCalibrator.cxx

#include "ITSCalibration/NoiseCalibrator.h"

#include "Framework/Logger.h"
#include "TFile.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace o2
{
namespace its
{

bool NoiseCalibrator::processTimeFrameClusters(gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                                               gsl::span<const unsigned char> const& patterns,
                                               gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  static int nTF = 0;
  LOG(info) << "Processing TF# " << nTF++;
  // extract hits
  auto pattIt = patterns.begin();
  for (const auto& rof : rofs) {
    mChipIDs.clear();
    int chipID = -1;
    std::vector<o2::itsmft::PixelData>* currChip = nullptr;
    auto clustersInFrame = rof.getROFData(clusters);
    for (const auto& c : clustersInFrame) {
      if (chipID != c.getSensorID()) { // data is sorted over chip IDs
        chipID = c.getSensorID();
        if (mChipHits.size() < mChipIDs.size() + 1) {
          mChipHits.resize(mChipIDs.size() + 1);
        }
        currChip = &mChipHits[mChipIDs.size()];
        mChipIDs.push_back(chipID);
      }
      auto pattID = c.getPatternID();
      o2::itsmft::ClusterPattern patt;
      auto row = c.getRow();
      auto col = c.getCol();
      if (mDict->getSize() == 0) {
        if (pattID == o2::itsmft::CompCluster::InvalidPatternID) {
          patt.acquirePattern(pattIt);
        } else {
          LOG(fatal) << "Clusters contain pattern IDs, but no dictionary is provided...";
        }
      } else if (pattID == o2::itsmft::CompCluster::InvalidPatternID) {
        patt.acquirePattern(pattIt);
      } else if (mDict->isGroup(pattID)) {
        patt.acquirePattern(pattIt);
        float xCOG = 0., zCOG = 0.;
        patt.getCOG(xCOG, zCOG); // for grouped patterns the reference pixel is at COG
        row -= round(xCOG);
        col -= round(zCOG);
      } else {
        patt = mDict->getPattern(pattID);
      }
      auto colSpan = patt.getColumnSpan();
      auto rowSpan = patt.getRowSpan();
      // Fast 1-pixel calibration
      if ((rowSpan == 1) && (colSpan == 1)) {
        currChip->emplace_back(row, col);
        continue;
      }
      if (m1pix) {
        continue;
      }
      for (int ir = 0; ir < rowSpan; ir++) {
        for (int ic = 0; ic < colSpan; ic++) {
          if (patt.isSet(ir, ic)) {
            currChip->emplace_back(row + ir, col + ic);
          }
        }
      }
    }
  }
  // distribute hits over the map
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(mNThreads)
#endif
  for (size_t ic = 0; ic < mChipIDs.size(); ic++) {
    int chipID = mChipIDs[ic];
    auto& hits = mChipHits[ic];
    for (const auto& hit : hits) {
      mNoiseMap.increaseNoiseCount(chipID, hit.getRowDirect(), hit.getCol());
    }
    hits.clear();
  }
  mNumberOfStrobes += rofs.size();
  return (mNumberOfStrobes > mMinROFs) ? true : false;
}

bool NoiseCalibrator::processTimeFrameDigits(gsl::span<const o2::itsmft::Digit> const& digits,
                                             gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  static int nTF = 0;
  LOG(info) << "Processing TF# " << nTF++;
  for (const auto& rof : rofs) {
    mChipIDs.clear();
    int chipID = -1;
    std::vector<o2::itsmft::PixelData>* currChip = nullptr;
    auto digitsInFrame = rof.getROFData(digits);
    for (const auto& dig : digitsInFrame) {
      if (chipID != dig.getChipIndex()) { // data is sorted over chip IDs
        chipID = dig.getChipIndex();
        if (mChipHits.size() < mChipIDs.size() + 1) {
          mChipHits.resize(mChipIDs.size() + 1);
        }
        currChip = &mChipHits[mChipIDs.size()];
        mChipIDs.push_back(chipID);
      }
      currChip->emplace_back(dig.getRow(), dig.getColumn());
    }
  }
  // distribute hits over the map
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(mNThreads)
#endif
  for (size_t ic = 0; ic < mChipIDs.size(); ic++) {
    int chipID = mChipIDs[ic];
    auto& hits = mChipHits[ic];
    for (const auto& hit : hits) {
      mNoiseMap.increaseNoiseCount(chipID, hit.getRowDirect(), hit.getCol());
    }
    hits.clear();
  }
  mNumberOfStrobes += rofs.size();
  return (mNumberOfStrobes > mMinROFs) ? true : false;
}

void NoiseCalibrator::finalize()
{
  LOG(info) << "Number of processed strobes is " << mNumberOfStrobes;
  mNoiseMap.applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes, mProbRelErr);
  mNoiseMap.print();
}

} // namespace its
} // namespace o2
