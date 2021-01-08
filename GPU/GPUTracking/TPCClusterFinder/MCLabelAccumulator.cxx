// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

MCLabelAccumulator::MCLabelAccumulator(GPUTPCClusterFinder& clusterer)
  : mIndexMap(clusterer.mPindexMap), mLabels(clusterer.mPinputLabels), mOutput(clusterer.mPlabelsByRow)
{
}

void MCLabelAccumulator::collect(const ChargePos& pos, Charge q)
{
  if (q == 0 || !engaged()) {
    return;
  }

  uint index = mIndexMap[pos];

  const auto& labels = mLabels->getLabels(index);

  for (const auto& label : labels) {
    int h = label.getRawValue() % mMaybeHasLabel.size();

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

void MCLabelAccumulator::commit(Row row, uint indexInRow, uint maxElemsPerBucket)
{
  if (indexInRow >= maxElemsPerBucket || !engaged()) {
    return;
  }

  auto& out = mOutput[row * maxElemsPerBucket + indexInRow];
  out.labels = std::move(mClusterLabels);
}
