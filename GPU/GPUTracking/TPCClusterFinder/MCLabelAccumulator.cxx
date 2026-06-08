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

/// \file MCLabelAccumulator.cxx
/// \author Felix Weiglhofer

#include "MCLabelAccumulator.h"

#include "GPUHostDataTypes.h"
#include "GPUTPCClusterFinder.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

MCLabelAccumulator::MCLabelAccumulator(GPUTPCClusterFinder& clusterer)
  : mIndexMap(clusterer.mPindexMap), mLabels(clusterer.mPinputLabels), mOutput(clusterer.mPlabelsByRow)
{
}

MCLabelAccumulator::~MCLabelAccumulator() = default;

void MCLabelAccumulator::collect(const CfChargePos& pos, float q)
{
  if (q == 0 || !engaged()) {
    return;
  }

  // Use -1 as sentinel to indicate a missing label.
  // Can't use zero charge, as HIP filter will zero existing digits.
  uint32_t index = mIndexMap[pos];
  if (index == uint32_t(-1)) {
    return;
  }

  const auto& labels = mLabels->getLabels(index);

  for (const auto& label : labels) {
    int32_t h = label.getRawValue() % mMaybeHasLabel.size();

    if (mMaybeHasLabel[h]) {
      auto lookup = std::find(mClusterLabels.begin(), mClusterLabels.end(), label);
      if (lookup != mClusterLabels.end()) {
        continue;
      }
    }

    mMaybeHasLabel[h] = true;
    mClusterLabels.emplace_back(label);
  }
}

void MCLabelAccumulator::collectTail(tpccf::Row row, tpccf::Pad pad, uint16_t tailStart, uint16_t tailEnd)
{
  if (!engaged()) {
    return;
  }

  const auto basePos = CfChargePos{row, pad, 0};

  for (uint16_t t = tailStart; t < tailEnd; t++) {
    const auto pos = basePos.delta({0, (int16_t)t});
    // Charge passed to collect() doesn't matter, collect() skips zero charges
    // But we know there's an interesting value, but it was zeroed in chargeMap by tail filter
    collect(pos, 1023.f);
  }
}

void MCLabelAccumulator::commit(Row row, uint32_t indexInRow, uint32_t maxElemsPerBucket)
{
  if (indexInRow >= maxElemsPerBucket || !engaged()) {
    return;
  }

  auto& out = mOutput[row];
  while (out.lock.test_and_set(std::memory_order_acquire)) {
    ;
  }
  if (out.data.size() <= indexInRow) {
    out.data.resize(indexInRow + 100); // Increase in steps of 100 at least to reduce number of resize operations
  }
  out.data[indexInRow].labels = std::move(mClusterLabels);
  out.lock.clear(std::memory_order_release);
}
