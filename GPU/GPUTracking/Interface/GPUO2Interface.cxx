// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2Interface.cxx
/// \author David Rohr

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUMemorySizeScalers.h"
#include "GPUOutputControl.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUParam.inc"
#include "GPUQA.h"
#include "GPUOutputControl.h"
#include <iostream>
#include <fstream>

using namespace o2::gpu;

#include "DataFormatsTPC/ClusterNative.h"

GPUO2Interface::GPUO2Interface() = default;

GPUO2Interface::~GPUO2Interface() { Deinitialize(); }

int GPUO2Interface::Initialize(const GPUO2InterfaceConfiguration& config)
{
  if (mInitialized) {
    return (1);
  }
  mConfig.reset(new GPUO2InterfaceConfiguration(config));
  mContinuous = mConfig->configEvent.continuousMaxTimeBin != 0;
  mRec.reset(GPUReconstruction::CreateInstance(mConfig->configDeviceBackend));
  if (mRec == nullptr) {
    GPUError("Error obtaining instance of GPUReconstruction");
    return 1;
  }
  mChain = mRec->AddChain<GPUChainTracking>(mConfig->configInterface.maxTPCHits, mConfig->configInterface.maxTRDTracklets);
  mChain->mConfigDisplay = &mConfig->configDisplay;
  mChain->mConfigQA = &mConfig->configQA;
  if (mConfig->configWorkflow.inputs.isSet(GPUDataTypes::InOutType::TPCRaw)) {
    mConfig->configEvent.needsClusterer = 1;
  }
  mRec->SetSettings(&mConfig->configEvent, &mConfig->configReconstruction, &mConfig->configProcessing, &mConfig->configWorkflow);
  mChain->SetCalibObjects(mConfig->configCalib);
  mOutputRegions.reset(new GPUTrackingOutputs);
  if (mConfig->configInterface.outputToExternalBuffers) {
    for (unsigned int i = 0; i < mOutputRegions->count(); i++) {
      mChain->SetSubOutputControl(i, &mOutputRegions->asArray()[i]);
    }
    GPUOutputControl dummy;
    dummy.set([](size_t size) -> void* {throw std::runtime_error("invalid output memory request, no common output buffer set"); return nullptr; });
    mRec->SetOutputControl(dummy);
  }

  if (mRec->Init()) {
    return (1);
  }
  if (!mRec->IsGPU() && mRec->GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    mRec->MemoryScalers()->factor *= 2;
  }
  mRec->MemoryScalers()->factor *= mConfig->configInterface.memoryBufferScaleFactor;
  mInitialized = true;
  return (0);
}

void GPUO2Interface::Deinitialize()
{
  if (mInitialized) {
    mRec->Finalize();
    mRec.reset();
  }
  mInitialized = false;
}

int GPUO2Interface::RunTracking(GPUTrackingInOutPointers* data, GPUInterfaceOutputs* outputs)
{
  if (!mInitialized) {
    return (1);
  }
  if (mConfig->configInterface.dumpEvents) {
    static int nEvent = 0;
    mChain->ClearIOPointers();
    mChain->mIOPtrs.clustersNative = data->clustersNative;
    mChain->mIOPtrs.tpcPackedDigits = data->tpcPackedDigits;
    mChain->mIOPtrs.tpcZS = data->tpcZS;

    char fname[1024];
    sprintf(fname, "event.%d.dump", nEvent);
    mChain->DumpData(fname);
    if (nEvent == 0) {
      mRec->DumpSettings();
    }
    nEvent++;
    if (mConfig->configInterface.dumpEvents >= 2) {
      return 0;
    }
  }

  mChain->mIOPtrs = *data;
  if (mConfig->configInterface.outputToExternalBuffers) {
    for (unsigned int i = 0; i < mOutputRegions->count(); i++) {
      if (outputs->asArray()[i].allocator) {
        mOutputRegions->asArray()[i].set(outputs->asArray()[i].allocator);
      } else if (outputs->asArray()[i].ptrBase) {
        mOutputRegions->asArray()[i].set(outputs->asArray()[i].ptrBase, outputs->asArray()[i].size);
      } else {
        mOutputRegions->asArray()[i].reset();
      }
    }
  }

  int retVal = mRec->RunChains();
  if (retVal == 2) {
    retVal = 0; // 2 signals end of event display, ignore
  }
  if (retVal) {
    mRec->ClearAllocatedMemory();
    return retVal;
  }
  if (mConfig->configQA.shipToQC) {
    outputs->qa.hist1 = &mChain->GetQA()->getHistograms1D();
    outputs->qa.hist2 = &mChain->GetQA()->getHistograms2D();
    outputs->qa.hist3 = &mChain->GetQA()->getHistograms1Dd();
  }
  *data = mChain->mIOPtrs;

  return 0;
}

void GPUO2Interface::Clear(bool clearOutputs) { mRec->ClearAllocatedMemory(clearOutputs); }

void GPUO2Interface::GetClusterErrors2(int row, float z, float sinPhi, float DzDs, short clusterState, float& ErrY2, float& ErrZ2) const
{
  mRec->GetParam().GetClusterErrors2(row, z, sinPhi, DzDs, ErrY2, ErrZ2);
  mRec->GetParam().UpdateClusterError2ByState(clusterState, ErrY2, ErrZ2);
}

int GPUO2Interface::registerMemoryForGPU(const void* ptr, size_t size)
{
  return mRec->registerMemoryForGPU(ptr, size);
}

int GPUO2Interface::unregisterMemoryForGPU(const void* ptr)
{
  return mRec->unregisterMemoryForGPU(ptr);
}
