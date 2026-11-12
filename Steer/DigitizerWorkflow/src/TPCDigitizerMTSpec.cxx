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

/// \file TPCDigitizerMTSpec.cxx
/// \brief A single TPC digitizer device that digitizes the sectors with several threads

#include "TPCDigitizerSpec.h"
#include "TPCSectorDigitization.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "SimConfig/DigiParams.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "TPCSimulation/Digitizer.h"
#include "CommonUtils/DebugStreamer.h"
#include <TChain.h>
#include <TROOT.h>
#include <TStopwatch.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace o2::framework;

namespace o2
{
namespace tpc
{

namespace
{
/// The state of one worker thread: a digitizer, its own hit chains and its own digit file
struct DigitizerWorker {
  DigitizerWorker(int lane, bool withMCTruth) : id(lane), writer(lane, withMCTruth) {}
  ~DigitizerWorker()
  {
    for (auto chain : chains) {
      delete chain;
    }
  }

  /// Digitize one sector and write it to the digit file of the worker
  void processSector(const steer::DigitizationContext& context, int sector, ULong64_t timeframe)
  {
    writer.beginSector(sector);
    digitizeSector(digitizer, context, chains, sector, timeframe, buffers, [this]() { writer.writeFlush(buffers); });
    writer.endSector();
  }

  int id = 0;
  Digitizer digitizer;
  std::vector<TChain*> chains;
  SectorFlushBuffers buffers;
  LaneDigitFileWriter writer;
  std::exception_ptr error;
};

std::string contextBinding(int sector)
{
  return "collisioncontext" + std::to_string(sector);
}
} // namespace

class TPCDPLDigitizerMTTask : public o2::base::BaseDPLDigitizer
{
 public:
  TPCDPLDigitizerMTTask(int nThreads, std::vector<int> sectors, int distortionType) : BaseDPLDigitizer(o2::base::InitServices::FIELD | o2::base::InitServices::GEOM), mSectors(std::move(sectors)), mNThreads(nThreads), mDistortionType(distortionType) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(info) << "Initializing multithreaded TPC digitization with " << mNThreads << " threads";

    const bool withMCTruth = o2::conf::DigiParams::Instance().mctruth;
    const auto triggeredMode = ic.options().get<bool>("TPCtriggered");
    mConditions.init(ic, mDistortionType);

    // the workers share no random numbers: each digitizer seeds every sector from the same base seed,
    // so the option --tpc-sector-seeding is always on here
    const ULong64_t sectorSeed = sectorSeedingBaseSeed();

    for (int i = 0; i < mNThreads; ++i) {
      auto& worker = mWorkers.emplace_back(std::make_unique<DigitizerWorker>(i, withMCTruth));
      worker->digitizer.setContinuousReadout(!triggeredMode);
      worker->digitizer.setDistortionScaleType(mDistortionType);
      worker->digitizer.setUseSectorSeeding(sectorSeed);
      // the main thread flushes the process-wide debug streamer after the workers
      worker->digitizer.setFlushDebugStreamer(false);
    }

    // the merger picks up every lane file in the directory, so files of an earlier run must not survive
    removeLaneDigitFiles([](int) { return true; });

    ROOT::EnableThreadSafety();
  }

  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
  {
    // the maps are set on the first digitizer; the others share them once they are final
    mConditions.finaliseCCDB(matcher, obj, mWorkers[0]->digitizer);
  }

  void run(framework::ProcessingContext& pc)
  {
    LOG(info) << "Processing multithreaded TPC digitization";

    auto& primary = mWorkers[0]->digitizer;
    if (auto vdrift = mConditions.update(pc, primary)) {
      for (auto& worker : mWorkers) {
        worker->digitizer.setVDrift(vdrift->vdrift);
        worker->digitizer.setTDriftOffset(vdrift->tdriftOffset);
      }
    }
    if (mDistortionType) {
      for (size_t i = 1; i < mWorkers.size(); ++i) {
        mWorkers[i]->digitizer.setSCDistortionsFrom(primary);
      }
    }
    if (!mSeedingShared) {
      // the rings depend only on the seed, so they are filled once and copied
      for (size_t i = 1; i < mWorkers.size(); ++i) {
        mWorkers[i]->digitizer.setSectorSeedingFrom(primary);
      }
      mSeedingShared = true;
    }

    if (mWriteGRP) {
      auto roMode = primary.isContinuousReadout() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
      LOG(info) << "TPC: Sending ROMode= " << (primary.isContinuousReadout() ? "Continuous" : "Triggered") << " to GRPUpdater";
      pc.outputs().snapshot(Output{"TPC", "ROMode", 0}, roMode);
      mWriteGRP = false;
    }
    if (mSectors.empty()) {
      return;
    }
    // the chunked digit files and the dynamic sector scheduling hold one timeframe per process
    if (mProcessedTimeframe) {
      LOG(fatal) << "TPC: the multithreaded digitizer supports only one timeframe per process";
    }
    mProcessedTimeframe = true;
    // every sector input carries the same collision context
    const auto contextRef = pc.inputs().get(contextBinding(mSectors[0]));
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>(contextRef);
    const ULong64_t timeframe = DataRefUtils::getHeader<o2::header::DataHeader*>(contextRef)->tfCounter;
    LOG(info) << "TPC: Processing " << mSectors.size() << " sectors with " << context->getEventRecords().size() << " collisions";

    TStopwatch timer;
    timer.Start();

    const size_t nWorkers = std::min(mWorkers.size(), mSectors.size());
    for (size_t i = 0; i < nWorkers; ++i) {
      context->initSimChains(o2::detectors::DetID::TPC, mWorkers[i]->chains);
    }

    // the sectors go to the workers dynamically
    std::atomic<size_t> nextSector{0};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < nWorkers; ++i) {
      threads.emplace_back([&, worker = mWorkers[i].get()]() {
        try {
          for (size_t is = nextSector++; is < mSectors.size(); is = nextSector++) {
            LOG(info) << "TPC: Worker " << worker->id << " processing sector " << mSectors[is];
            worker->processSector(*context, mSectors[is], timeframe);
          }
        } catch (...) {
          worker->error = std::current_exception();
          nextSector = mSectors.size(); // let the other workers stop early
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    if (auto* streamer = o2::utils::DebugStreamer::instance()) {
      streamer->flush();
    }
    for (size_t i = 0; i < nWorkers; ++i) {
      if (mWorkers[i]->error) {
        std::rethrow_exception(mWorkers[i]->error);
      }
    }

    timer.Stop();
    LOG(info) << "TPC: Digitization took " << timer.CpuTime() << "s CPU, " << timer.RealTime() << "s wall";
  }

 private:
  std::vector<std::unique_ptr<DigitizerWorker>> mWorkers;
  std::vector<int> mSectors; ///< the sectors to digitize, one collision context input each
  TPCDigitizerConditions mConditions;
  bool mSeedingShared = false;      ///< whether the workers got the random rings of the first one
  bool mProcessedTimeframe = false; ///< whether a timeframe was digitized
  int mNThreads = 1;
  int mDistortionType = 0;
  bool mWriteGRP = true;
};

o2::framework::DataProcessorSpec getTPCDigitizerMTSpec(int nThreads, std::vector<int> const& sectors, int distortionType)
{
  // the reader sends the collision context once per sector, so the device subscribes to all of them
  std::vector<InputSpec> inputs;
  for (auto sector : sectors) {
    inputs.emplace_back(contextBinding(sector), "SIM", "COLLISIONCONTEXT", static_cast<o2::header::DataHeader::SubSpecificationType>(sector), Lifetime::Timeframe);
  }
  TPCDigitizerConditions::addInputs(inputs, distortionType);
  std::vector<OutputSpec> outputs{{"TPC", "ROMode", 0, Lifetime::Timeframe}};

  // the name must not match "TPCDigitizer.*", whose completion policy would process the sectors one by one
  return DataProcessorSpec{
    "tpc-digitizer-mt",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCDPLDigitizerMTTask>(nThreads, sectors, distortionType)},
    TPCDigitizerConditions::options()};
}

} // namespace tpc
} // namespace o2
