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

/// @file   NoiseCalibrator.h

#ifndef O2_ITS_NOISECALIBRATOR
#define O2_ITS_NOISECALIBRATOR

#include <string>
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "gsl/span"

namespace o2
{

namespace itsmft
{
class CompClusterExt;
class ROFRecord;
} // namespace itsmft

namespace its
{

class NoiseCalibrator
{
 public:
  static constexpr int NChips = o2::itsmft::ChipMappingITS::getNChips();

  NoiseCalibrator() = default;
  NoiseCalibrator(bool one, float prob, float relErr = 0.2) : m1pix(one), mProbabilityThreshold(prob), mProbRelErr(relErr)
  {
    mMinROFs = 1.1 * o2::itsmft::NoiseMap::getMinROFs(prob, relErr);
    LOGP(info, "Expect at least {} ROFs needed to apply threshold {} with relative error {}", mMinROFs, mProbabilityThreshold, mProbRelErr);
  }
  ~NoiseCalibrator() = default;

  bool processTimeFrameClusters(gsl::span<const o2::itsmft::CompClusterExt> const& clusters,
                                gsl::span<const unsigned char> const& patterns,
                                gsl::span<const o2::itsmft::ROFRecord> const& rofs);

  bool processTimeFrameDigits(gsl::span<const o2::itsmft::Digit> const& digits,
                              gsl::span<const o2::itsmft::ROFRecord> const& rofs);

  void addMap(const o2::itsmft::NoiseMap& extMap);

  void finalize(float cutIB = -1.);

  void setNThreads(int n) { mNThreads = n > 0 ? n : 1; }

  void setMinROFs(long n) { mMinROFs = n; }
  long getMinROFs() const { return mMinROFs; }

  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }

  const o2::itsmft::NoiseMap& getNoiseMap() const { return mNoiseMap; }

  void setInstanceID(int i) { mInstanceID = i; }
  void setNInstances(int n) { mNInstances = n; }
  auto getInstanceID() const { return mInstanceID; }
  auto getNInstances() const { return mNInstances; }
  auto getNStrobes() const { return mNumberOfStrobes; }
  auto setNStrobes(unsigned int s) { mNumberOfStrobes = s; }

  void reset();

 private:
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  o2::itsmft::NoiseMap mNoiseMap{NChips};
  float mProbabilityThreshold = 3e-6f;
  float mProbRelErr = 0.2; // relative error on channel noise to apply the threshold
  long mMinROFs = 0;
  unsigned int mNumberOfStrobes = 0;
  bool m1pix = true;
  int mNThreads = 1;
  int mInstanceID = 0; // pipeline instance
  int mNInstances = 1; // total number of pipelines
  std::vector<int> mChipIDs;
  std::array<std::vector<int>, NChips> mChipHits;
};

} // namespace its
} // namespace o2

#endif /* O2_ITS_NOISECALIBRATOR */
