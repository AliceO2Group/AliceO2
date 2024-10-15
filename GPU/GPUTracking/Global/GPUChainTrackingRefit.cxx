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

/// \file GPUChainTrackingRefit.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"

using namespace GPUCA_NAMESPACE::gpu;

int32_t GPUChainTracking::RunRefit()
{
#ifdef GPUCA_HAVE_O2HEADERS
  bool doGPU = GetRecoStepsGPU() & RecoStep::Refit;
  GPUTrackingRefitProcessor& Refit = processors()->trackingRefit;
  GPUTrackingRefitProcessor& RefitShadow = doGPU ? processorsShadow()->trackingRefit : Refit;

  const auto& threadContext = GetThreadContext();
  (void)threadContext;
  SetupGPUProcessor(&Refit, false);
  RefitShadow.SetPtrsFromGPUConstantMem(processorsShadow(), doGPU ? &processorsDevice()->param : nullptr);
  RefitShadow.SetPropagator(doGPU ? processorsShadow()->calibObjects.o2Propagator : GetO2Propagator());
  RefitShadow.mPTracks = (doGPU ? processorsShadow() : processors())->tpcMerger.OutputTracks();
  WriteToConstantMemory(RecoStep::Refit, (char*)&processors()->trackingRefit - (char*)processors(), &RefitShadow, sizeof(RefitShadow), 0);
  //TransferMemoryResourcesToGPU(RecoStep::Refit, &Refit, 0);
  if (param().rec.trackingRefitGPUModel) {
    runKernel<GPUTrackingRefitKernel, GPUTrackingRefitKernel::mode0asGPU>(GetGrid(mIOPtrs.nMergedTracks, 0));
  } else {
    runKernel<GPUTrackingRefitKernel, GPUTrackingRefitKernel::mode1asTrackParCov>(GetGrid(mIOPtrs.nMergedTracks, 0));
  }
  //TransferMemoryResourcesToHost(RecoStep::Refit, &Refit, 0);
  SynchronizeStream(0);
#endif
  return 0;
}
