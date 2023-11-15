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
#include "GPUChainITS.h"
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
#include <thread>
#include <optional>
#include <mutex>

using namespace o2::gpu;

#include "DataFormatsTPC/ClusterNative.h"

namespace o2::gpu
{
struct GPUO2Interface_processingContext {
  std::unique_ptr<GPUReconstruction> mRec;
  GPUChainTracking* mChain = nullptr;
  std::unique_ptr<GPUTrackingOutputs> mOutputRegions;
};

struct GPUO2Interface_Internals {
  std::unique_ptr<std::thread> pipelineThread;
};
} // namespace o2::gpu

GPUO2Interface::GPUO2Interface() : mInternals(new GPUO2Interface_Internals){};

GPUO2Interface::~GPUO2Interface() { Deinitialize(); }

int GPUO2Interface::Initialize(const GPUO2InterfaceConfiguration& config)
{
  if (mNContexts) {
    return (1);
  }
  mConfig.reset(new GPUO2InterfaceConfiguration(config));
  mNContexts = mConfig->configProcessing.doublePipeline ? 2 : 1;
  mCtx.reset(new GPUO2Interface_processingContext[mNContexts]);
  if (mConfig->configWorkflow.inputs.isSet(GPUDataTypes::InOutType::TPCRaw)) {
    mConfig->configGRP.needsClusterer = 1;
  }
  if (mConfig->configWorkflow.inputs.isSet(GPUDataTypes::InOutType::TPCCompressedClusters)) {
    mConfig->configGRP.doCompClusterDecode = 1;
  }
  for (unsigned int i = 0; i < mNContexts; i++) {
    if (i) {
      mConfig->configDeviceBackend.master = mCtx[0].mRec.get();
    }
    mCtx[i].mRec.reset(GPUReconstruction::CreateInstance(mConfig->configDeviceBackend));
    mConfig->configDeviceBackend.master = nullptr;
    if (mCtx[i].mRec == nullptr) {
      GPUError("Error obtaining instance of GPUReconstruction");
      mNContexts = 0;
      mCtx.reset(nullptr);
      return 1;
    }
  }
  for (unsigned int i = 0; i < mNContexts; i++) {
    mCtx[i].mChain = mCtx[i].mRec->AddChain<GPUChainTracking>(mConfig->configInterface.maxTPCHits, mConfig->configInterface.maxTRDTracklets);
    if (i) {
      mCtx[i].mChain->SetQAFromForeignChain(mCtx[0].mChain);
    }
    mCtx[i].mChain->mConfigDisplay = &mConfig->configDisplay;
    mCtx[i].mChain->mConfigQA = &mConfig->configQA;
    mCtx[i].mRec->SetSettings(&mConfig->configGRP, &mConfig->configReconstruction, &mConfig->configProcessing, &mConfig->configWorkflow);
    mCtx[i].mChain->SetCalibObjects(mConfig->configCalib);

    if (i == 0 && mConfig->configWorkflow.steps.isSet(GPUDataTypes::RecoStep::ITSTracking)) {
      mChainITS = mCtx[i].mRec->AddChain<GPUChainITS>();
    }

    mCtx[i].mOutputRegions.reset(new GPUTrackingOutputs);
    if (mConfig->configInterface.outputToExternalBuffers) {
      for (unsigned int j = 0; j < mCtx[i].mOutputRegions->count(); j++) {
        mCtx[i].mChain->SetSubOutputControl(j, &mCtx[i].mOutputRegions->asArray()[j]);
      }
      GPUOutputControl dummy;
      dummy.set([](size_t size) -> void* {throw std::runtime_error("invalid output memory request, no common output buffer set"); return nullptr; });
      mCtx[i].mRec->SetOutputControl(dummy);
    }
  }
  for (unsigned int i = 0; i < mNContexts; i++) {
    if (i == 0 && mCtx[i].mRec->Init()) {
      mNContexts = 0;
      mCtx.reset(nullptr);
      return (1);
    }
    if (!mCtx[i].mRec->IsGPU() && mCtx[i].mRec->GetProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
      mCtx[i].mRec->MemoryScalers()->factor *= 2;
    }
  }
  if (mConfig->configProcessing.doublePipeline) {
    mInternals->pipelineThread.reset(new std::thread([this]() { mCtx[0].mRec->RunPipelineWorker(); }));
  }
  return (0);
}

void GPUO2Interface::Deinitialize()
{
  if (mNContexts) {
    if (mConfig->configProcessing.doublePipeline) {
      mCtx[0].mRec->TerminatePipelineWorker();
      mInternals->pipelineThread->join();
    }
    for (unsigned int i = 0; i < mNContexts; i++) {
      mCtx[i].mRec->Finalize();
    }
    mCtx[0].mRec->Exit();
    for (int i = mNContexts - 1; i >= 0; i--) {
      mCtx[i].mRec.reset();
    }
  }
  mNContexts = 0;
}

void GPUO2Interface::DumpEvent(int nEvent, GPUTrackingInOutPointers* data)
{
  mCtx[0].mChain->ClearIOPointers();
  mCtx[0].mChain->mIOPtrs = *data;
  char fname[1024];
  snprintf(fname, 1024, "event.%d.dump", nEvent);
  mCtx[0].mChain->DumpData(fname);
  if (nEvent == 0) {
#ifdef GPUCA_BUILD_QA
    if (mConfig->configProcessing.runMC) {
      mCtx[0].mChain->ForceInitQA();
      snprintf(fname, 1024, "mc.%d.dump", nEvent);
      mCtx[0].mChain->GetQA()->UpdateChain(mCtx[0].mChain);
      mCtx[0].mChain->GetQA()->DumpO2MCData(fname);
    }
#endif
  }
}

void GPUO2Interface::DumpSettings()
{
  mCtx[0].mChain->DoQueuedUpdates(-1);
  mCtx[0].mRec->DumpSettings();
}

int GPUO2Interface::RunTracking(GPUTrackingInOutPointers* data, GPUInterfaceOutputs* outputs, unsigned int iThread, GPUInterfaceInputUpdate* inputUpdateCallback)
{
  if (mNContexts <= iThread) {
    return (1);
  }

  mCtx[iThread].mChain->mIOPtrs = *data;

  auto setOutputs = [this, iThread](GPUInterfaceOutputs* outputs) {
    if (mConfig->configInterface.outputToExternalBuffers) {
      for (unsigned int i = 0; i < mCtx[iThread].mOutputRegions->count(); i++) {
        if (outputs->asArray()[i].allocator) {
          mCtx[iThread].mOutputRegions->asArray()[i].set(outputs->asArray()[i].allocator);
        } else if (outputs->asArray()[i].ptrBase) {
          mCtx[iThread].mOutputRegions->asArray()[i].set(outputs->asArray()[i].ptrBase, outputs->asArray()[i].size);
        } else {
          mCtx[iThread].mOutputRegions->asArray()[i].reset();
        }
      }
    }
  };

  auto inputWaitCallback = [this, iThread, inputUpdateCallback, &data, &outputs, &setOutputs]() {
    GPUTrackingInOutPointers* updatedData;
    GPUInterfaceOutputs* updatedOutputs;
    if (inputUpdateCallback->callback) {
      inputUpdateCallback->callback(updatedData, updatedOutputs);
      mCtx[iThread].mChain->mIOPtrs = *updatedData;
      outputs = updatedOutputs;
      data = updatedData;
      setOutputs(outputs);
    }
    if (inputUpdateCallback->notifyCallback) {
      inputUpdateCallback->notifyCallback();
    }
  };

  if (inputUpdateCallback) {
    mCtx[iThread].mChain->SetFinalInputCallback(inputWaitCallback);
  } else {
    mCtx[iThread].mChain->SetFinalInputCallback(nullptr);
  }
  if (!inputUpdateCallback || !inputUpdateCallback->callback) {
    setOutputs(outputs);
  }

  int retVal = mCtx[iThread].mRec->RunChains();
  if (retVal == 2) {
    retVal = 0; // 2 signals end of event display, ignore
  }
  if (mConfig->configQA.shipToQC && mCtx[iThread].mChain->QARanForTF()) {
    outputs->qa.hist1 = &mCtx[iThread].mChain->GetQA()->getHistograms1D();
    outputs->qa.hist2 = &mCtx[iThread].mChain->GetQA()->getHistograms2D();
    outputs->qa.hist3 = &mCtx[iThread].mChain->GetQA()->getHistograms1Dd();
    outputs->qa.hist4 = &mCtx[iThread].mChain->GetQA()->getGraphs();
    outputs->qa.newQAHistsCreated = true;
  }
  *data = mCtx[iThread].mChain->mIOPtrs;

  return retVal;
}

void GPUO2Interface::Clear(bool clearOutputs, unsigned int iThread) { mCtx[iThread].mRec->ClearAllocatedMemory(clearOutputs); }

int GPUO2Interface::registerMemoryForGPU(const void* ptr, size_t size)
{
  return mCtx[0].mRec->registerMemoryForGPU(ptr, size);
}

int GPUO2Interface::unregisterMemoryForGPU(const void* ptr)
{
  return mCtx[0].mRec->unregisterMemoryForGPU(ptr);
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

int GPUO2Interface::UpdateCalibration(const GPUCalibObjectsConst& newCalib, const GPUNewCalibValues& newVals, unsigned int iThread)
{
  for (unsigned int i = 0; i < mNContexts; i++) {
    mCtx[i].mChain->SetUpdateCalibObjects(newCalib, newVals);
  }
  return 0;
}

void GPUO2Interface::setErrorCodeOutput(std::vector<std::array<unsigned int, 4>>* v)
{
  for (unsigned int i = 0; i < mNContexts; i++) {
    mCtx[i].mRec->setErrorCodeOutput(v);
  }
}

void GPUO2Interface::GetITSTraits(o2::its::TrackerTraits*& trackerTraits, o2::its::VertexerTraits*& vertexerTraits, o2::its::TimeFrame*& timeFrame)
{
  trackerTraits = mChainITS->GetITSTrackerTraits();
  vertexerTraits = mChainITS->GetITSVertexerTraits();
  timeFrame = mChainITS->GetITSTimeframe();
}
