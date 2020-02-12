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

#include "GPUTPCClusterFinder.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace GPUCA_NAMESPACE::gpu;

MCLabelAccumulator::MCLabelAccumulator(GPUTPCClusterFinder& clusterer)
  : mIndexMap(clusterer.mPindexMap), mLabels(clusterer.mPinputLabels), mOutput(clusterer.mPclusterLabels)
{
}

void MCLabelAccumulator::collect(const ChargePos& pos, Charge q)
{
  if (q == 0 || !engaged()) {
    return;
  }

  DigitID index = mIndexMap[pos];

  auto labels = mLabels->getLabels(index);

  mClusterLabels.insert(labels.begin(), labels.end());
}

void MCLabelAccumulator::commit(Row row, ClusterID index)
{
  if (!engaged()) {
    return;
  }

  for (const auto& label : mClusterLabels) {
    mOutput->labels[row]->addElement(index, label);
  }
}
