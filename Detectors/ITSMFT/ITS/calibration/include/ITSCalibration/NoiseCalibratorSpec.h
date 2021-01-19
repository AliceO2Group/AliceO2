// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   NoiseCalibratorSpec.h

#ifndef O2_ITS_NOISECALIBRATOR
#define O2_ITS_NOISECALIBRATOR

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"

using namespace o2::framework;

namespace o2
{

namespace its
{

class NoiseCalibrator : public Task
{
 public:
  NoiseCalibrator() = default;
  ~NoiseCalibrator() override = default;

  void setProbabilityThreshold(float t) { mProbabilityThreshold = t; }
  void setOnePix(bool onepix = true) { m1pix = onepix; }
  void setUseCCDB(bool use = true) { mUseCCDB = use; }
  void processTimeFrame(gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                        gsl::span<const unsigned char> const& patterns,
                        gsl::span<const o2::itsmft::ROFRecord> const& rofs);
  void registerNoiseMap();

  //DPL-oriented functions
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  o2::itsmft::NoiseMap mNoiseMap{24120};
  float mProbabilityThreshold = 3e-6f;
  unsigned int mNumberOfStrobes = 0;
  bool mUseCCDB = false;
  bool m1pix = true;
};

/// create a processor spec
/// run ITS noise calibration
DataProcessorSpec getNoiseCalibratorSpec();

} // namespace its
} // namespace o2

#endif /* O2_ITS_CLUSTERERDPL */
