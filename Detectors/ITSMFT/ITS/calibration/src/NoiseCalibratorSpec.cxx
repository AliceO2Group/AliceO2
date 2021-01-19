// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseCalibratorSpec.cxx

#include "ITSCalibration/NoiseCalibratorSpec.h"

#include "FairLogger.h"
#include "TFile.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsITSMFT/ClusterPattern.h"

using namespace o2::framework;

namespace o2
{
namespace its
{
void NoiseCalibrator::processTimeFrame(gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                                       gsl::span<const unsigned char> const& patterns,
                                       gsl::span<const o2::itsmft::ROFRecord> const& rofs)
{
  static int nTF = 0;
  LOG(INFO) << "Processing TF# " << nTF++;

  auto pattIt = patterns.cbegin();
  for (const auto& rof : rofs) {
    auto clustersInFrame = rof.getROFData(clusters);
    for (const auto& c : clustersInFrame) {
      if (c.getPatternID() != o2::itsmft::CompCluster::InvalidPatternID) {
        // For the noise calibration, we use "pass1" clusters...
        continue;
      }
      o2::itsmft::ClusterPattern patt(pattIt);

      auto id = c.getSensorID();
      auto row = c.getRow();
      auto col = c.getCol();
      auto colSpan = patt.getColumnSpan();
      auto rowSpan = patt.getRowSpan();

      // Fast 1-pixel calibration
      if ((rowSpan == 1) && (colSpan == 1)) {
        mNoiseMap.increaseNoiseCount(id, row, col);
        continue;
      }
      if (m1pix) {
        continue;
      }

      // All-pixel calibration
      auto nBits = rowSpan * colSpan;
      int ic = 0, ir = 0;
      for (unsigned int i = 2; i < patt.getUsedBytes() + 2; i++) {
        unsigned char tempChar = patt.getByte(i);
        int s = 128; // 0b10000000
        while (s > 0) {
          if ((tempChar & s) != 0) {
            mNoiseMap.increaseNoiseCount(id, row + ir, col + ic);
          }
          ic++;
          s >>= 1;
          if ((ir + 1) * ic == nBits) {
            break;
          }
          if (ic == colSpan) {
            ic = 0;
            ir++;
          }
        }
        if ((ir + 1) * ic == nBits) {
          break;
        }
      }
    }
  }
  mNumberOfStrobes += rofs.size();
}

void NoiseCalibrator::init(InitContext& ic)
{
  mUseCCDB = ic.options().get<bool>("use-ccdb");
  LOG(INFO) << "Registration in CCDB: " << mUseCCDB;
  m1pix = ic.options().get<bool>("1pix-only");
  LOG(INFO) << "Fast 1=pixel calibration: " << m1pix;
  mProbabilityThreshold = ic.options().get<float>("prob-threshold");
  LOG(INFO) << "Setting the probability threshold to " << mProbabilityThreshold;
}

void NoiseCalibrator::run(ProcessingContext& pc)
{
  const auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");

  processTimeFrame(compClusters, patterns, rofs);
}

void NoiseCalibrator::registerNoiseMap()
{
  LOG(INFO) << "Number of processed strobes is " << mNumberOfStrobes;
  mNoiseMap.applyProbThreshold(mProbabilityThreshold, mNumberOfStrobes);

  if (!mUseCCDB) {
    TFile out("noise.root", "new");
    if (!out.IsOpen()) {
      LOG(ERROR) << "The output file already exists !";
      return;
    }
    out.WriteObject(&mNoiseMap, "Noise");
    out.Close();
  } else {
    LOG(INFO) << "Registering the noise map in CCDB...";
  }
}

void NoiseCalibrator::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  registerNoiseMap();
}

DataProcessorSpec getNoiseCalibratorSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "its-noise-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<NoiseCalibrator>()},
    Options{
      {"use-ccdb", VariantType::Bool, false, {"Register the noise map in CCDB"}},
      {"1pix-only", VariantType::Bool, false, {"Fast 1-pixel calibration only"}},
      {"prob-threshold", VariantType::Float, 3.e-6f, {"Probability threshold for noisy pixels"}}}};
}

} // namespace its
} // namespace o2
