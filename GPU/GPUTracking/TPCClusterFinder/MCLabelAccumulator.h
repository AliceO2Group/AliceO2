// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MCLabelAccumulator.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_MC_LABEL_ACCUMULATOR_H
#define O2_GPU_MC_LABEL_ACCUMULATOR_H

#include "clusterFinderDefs.h"
#include "Array2D.h"
#include "CPULabelContainer.h"
#include "GPUHostDataTypes.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <unordered_set>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTPCClusterFinder;
using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

class MCLabelAccumulator
{

 public:
  MCLabelAccumulator(GPUTPCClusterFinder&);

  void collect(const ChargePos&, Charge);

  bool engaged() const { return mLabels != nullptr && mOutput != nullptr; }

  uint commit(Row);

 private:
  Array2D<const DigitID> mIndexMap;
  const MCLabelContainer* mLabels = nullptr;
  GPUTPCClusterMCSector* mOutput = nullptr;
  std::unordered_set<o2::MCCompLabel> mClusterLabels;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
