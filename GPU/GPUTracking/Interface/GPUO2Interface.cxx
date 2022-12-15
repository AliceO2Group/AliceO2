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
#include "TPCPadGainCalib.h"
#include "CalibdEdxContainer.h"
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
  mContinuous = mConfig->configGRP.continuousMaxTimeBin != 0;
  mRec.reset(GPUReconstruction::CreateInstance(mConfig->configDeviceBackend));
  if (mRec == nullptr) {
    GPUError("Error obtaining instance of GPUReconstruction");
    return 1;
  }
  mChain = mRec->AddChain<GPUChainTracking>(mConfig->configInterface.maxTPCHits, mConfig->configInterface.maxTRDTracklets);
  mChain->mConfigDisplay = &mConfig->configDisplay;
  mChain->mConfigQA = &mConfig->configQA;
  if (mConfig->configWorkflow.inputs.isSet(GPUDataTypes::InOutType::TPCRaw)) {
    mConfig->configGRP.needsClusterer = 1;
  }
  mRec->SetSettings(&mConfig->configGRP, &mConfig->configReconstruction, &mConfig->configProcessing, &mConfig->configWorkflow);
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
    if (mConfig->configProcessing.doublePipeline) {
      throw std::runtime_error("Cannot dump events in double pipeline mode");
    }
    static int nEvent = 0;
    mChain->DoQueuedCalibUpdates(-1);
    mChain->ClearIOPointers();
    mChain->mIOPtrs = *data;

    char fname[1024];
    snprintf(fname, 1024, "event.%d.dump", nEvent);
    mChain->DumpData(fname);
    if (nEvent == 0) {
      mRec->DumpSettings();
#ifdef GPUCA_BUILD_QA
      if (mConfig->configProcessing.runMC) {
        mChain->ForceInitQA();
        snprintf(fname, 1024, "mc.%d.dump", nEvent);
        mChain->GetQA()->DumpO2MCData(fname);
      }
#endif
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
  if (mConfig->configQA.shipToQC && mChain->QARanForTF()) {
    outputs->qa.hist1 = &mChain->GetQA()->getHistograms1D();
    outputs->qa.hist2 = &mChain->GetQA()->getHistograms2D();
    outputs->qa.hist3 = &mChain->GetQA()->getHistograms1Dd();
    outputs->qa.newQAHistsCreated = true;
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

std::unique_ptr<TPCPadGainCalib> GPUO2Interface::getPadGainCalibDefault()
{
  return std::make_unique<TPCPadGainCalib>();
}

std::unique_ptr<TPCPadGainCalib> GPUO2Interface::getPadGainCalib(const o2::tpc::CalDet<float>& in)
{
  return std::make_unique<TPCPadGainCalib>(in);
}

std::unique_ptr<o2::tpc::CalibdEdxContainer> GPUO2Interface::getCalibdEdxContainerDefault()
{
  return std::make_unique<o2::tpc::CalibdEdxContainer>();
}

int GPUO2Interface::UpdateCalibration(const GPUCalibObjectsConst& newCalib, const GPUNewCalibValues& newVals)
{
  mChain->SetUpdateCalibObjects(newCalib, newVals);
  return 0;
}
