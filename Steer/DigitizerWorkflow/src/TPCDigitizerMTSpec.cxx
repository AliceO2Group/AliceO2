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
#include "DetectorsBase/Detector.h"
#include "SimConfig/DigiParams.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "TPCSimulation/Digitizer.h"
#include "CommonUtils/DebugStreamer.h"
#include "CommonUtils/FileSystemUtils.h"
#include <TChain.h>
#include <TFile.h>
#include <TROOT.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TTree.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
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
  int id = 0;
  Digitizer digitizer;
  std::vector<TChain*> chains;
  SectorFlushBuffers buffers;
  std::exception_ptr error;

  ~DigitizerWorker()
  {
    for (auto chain : chains) {
      delete chain;
    }
  }

  std::string fileName() const { return "tpc_driftime_digits_lane" + std::to_string(id) + ".root"; }

  /// Digitize one sector and write it to the worker's digit file, in the format of the chunked writer of the lane spec
  void processSector(const steer::DigitizationContext& context, int sector, ULong64_t timeframe, bool withMCTruth)
  {
    std::unique_ptr<TFile> file(TFile::Open(fileName().c_str(), "UPDATE"));
    if (!file || file->IsZombie()) {
      throw std::runtime_error("TPC: cannot open " + fileName());
    }
    auto tree = new TTree(std::to_string(sector).c_str(), "o2sim");
    tree->SetDirectory(file.get()); // owned by the file
    size_t flushCounter = 0;
    auto onFlush = [&]() {
      ++flushCounter;
      auto fill = [&tree](const std::string& name, auto* data) {
        auto br = o2::base::getOrMakeBranch(*tree, name.c_str(), data);
        br->Fill();
        br->ResetAddress();
      };
      fill("TPCDigit_" + std::to_string(sector), &buffers.digits);
      if (withMCTruth) {
        fill("TPCDigitMCTruth_" + std::to_string(sector), &buffers.labels);
      }
      fill("TPCCommonMode_" + std::to_string(sector), &buffers.commonMode);
    };
    digitizeSector(digitizer, context, chains, sector, timeframe, buffers, onFlush);
    tree->SetEntries(flushCounter);
    file->Write("", TObject::kOverwrite);
    file->Close();
  }
};
} // namespace

class TPCDPLDigitizerMTTask : public o2::base::BaseDPLDigitizer
{
 public:
  TPCDPLDigitizerMTTask(int nThreads, int distortionType) : BaseDPLDigitizer(o2::base::InitServices::FIELD | o2::base::InitServices::GEOM), mNThreads(nThreads), mDistortionType(distortionType) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(info) << "Initializing multithreaded TPC digitization with " << mNThreads << " threads";

    mWithMCTruth = o2::conf::DigiParams::Instance().mctruth;
    const auto triggeredMode = ic.options().get<bool>("TPCtriggered");
    mConditions.init(ic, mDistortionType);

    // the workers share no random numbers: each digitizer seeds every sector from DigiParams.seed
    const auto seed = o2::conf::DigiParams::Instance().seed;
    if (seed == 0) {
      LOG(warning) << "TPC sector seeding with DigiParams.seed=0 is not reproducible";
    }
    // with seed 0, gRandom was seeded randomly in the base class
    const ULong64_t sectorSeed = seed != 0 ? seed : gRandom->Integer(kMaxUInt);

    for (int i = 0; i < mNThreads; ++i) {
      auto& worker = mWorkers.emplace_back(std::make_unique<DigitizerWorker>());
      worker->id = i;
      worker->digitizer.setContinuousReadout(!triggeredMode);
      worker->digitizer.setDistortionScaleType(mDistortionType);
      worker->digitizer.setUseSectorSeeding(sectorSeed);
      // the main thread flushes the process-wide debug streamer after the workers
      worker->digitizer.setFlushDebugStreamer(false);
    }

    // the merger picks up every lane file in the directory, so files of an earlier run must not survive
    for (const auto& name : o2::utils::listFiles("tpc_driftime_digits_lane.*.root$")) {
      std::filesystem::remove(name);
    }

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
    mConditions.update(pc, primary, [this](float vdrift, float tdriftOffset) {
      for (auto& worker : mWorkers) {
        worker->digitizer.setVDrift(vdrift);
        worker->digitizer.setTDriftOffset(tdriftOffset);
      }
    });
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

    // the sectors of this timeframe; all inputs carry the same collision context
    std::vector<int> sectors;
    std::optional<DataRef> contextRef;
    for (auto it = pc.inputs().begin(), end = pc.inputs().end(); it != end; ++it) {
      for (auto const& inputref : it.parts()) {
        if (inputref.spec->lifetime == o2::framework::Lifetime::Condition) {
          continue;
        }
        auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(inputref);
        if (sectorHeader == nullptr) {
          LOG(error) << "TPC: Sector header missing, skipping input";
          continue;
        }
        const int sector = sectorHeader->sector();
        if (sector < 0 || sector >= TPCSectorHeader::NSectors) {
          throw std::runtime_error("Digitizer can only work on single sectors");
        }
        sectors.push_back(sector);
        if (!contextRef) {
          contextRef = inputref;
        }
      }
    }

    if (mWriteGRP) {
      auto roMode = primary.isContinuousReadout() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
      LOG(info) << "TPC: Sending ROMode= " << (primary.isContinuousReadout() ? "Continuous" : "Triggered") << " to GRPUpdater";
      pc.outputs().snapshot(Output{"TPC", "ROMode", 0}, roMode);
      mWriteGRP = false;
    }
    if (!contextRef) {
      return;
    }
    // the chunked digit files and the dynamic sector scheduling hold one timeframe per process
    if (mProcessedTimeframe) {
      LOG(fatal) << "TPC: the multithreaded digitizer supports only one timeframe per process";
    }
    mProcessedTimeframe = true;
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>(*contextRef);
    const ULong64_t timeframe = DataRefUtils::getHeader<o2::header::DataHeader*>(*contextRef)->tfCounter;
    LOG(info) << "TPC: Processing " << sectors.size() << " sectors with " << context->getEventRecords().size() << " collisions";

    TStopwatch timer;
    timer.Start();

    const size_t nWorkers = std::min(mWorkers.size(), sectors.size());
    for (size_t i = 0; i < nWorkers; ++i) {
      context->initSimChains(o2::detectors::DetID::TPC, mWorkers[i]->chains);
    }

    // the sectors go to the workers dynamically
    std::atomic<size_t> nextSector{0};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < nWorkers; ++i) {
      threads.emplace_back([&, worker = mWorkers[i].get()]() {
        try {
          for (size_t is = nextSector++; is < sectors.size(); is = nextSector++) {
            LOG(info) << "TPC: Worker " << worker->id << " processing sector " << sectors[is];
            worker->processSector(*context, sectors[is], timeframe, mWithMCTruth);
          }
        } catch (...) {
          worker->error = std::current_exception();
          nextSector = sectors.size(); // let the other workers stop early
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
  TPCDigitizerConditions mConditions;
  bool mSeedingShared = false;      ///< whether the workers got the random rings of the first one
  bool mProcessedTimeframe = false; ///< whether a timeframe was digitized
  int mNThreads = 1;
  int mDistortionType = 0;
  bool mWriteGRP = true;
  bool mWithMCTruth = true;
};

o2::framework::DataProcessorSpec getTPCDigitizerMTSpec(int nThreads, std::vector<int> const& sectors, int distortionType)
{
  std::vector<InputSpec> inputs;
  for (auto sector : sectors) {
    inputs.emplace_back("collisioncontext" + std::to_string(sector), "SIM", "COLLISIONCONTEXT", static_cast<o2::header::DataHeader::SubSpecificationType>(sector), Lifetime::Timeframe);
  }
  TPCDigitizerConditions::addInputs(inputs, distortionType);
  std::vector<OutputSpec> outputs{{"TPC", "ROMode", 0, Lifetime::Timeframe}};

  // the name must not match "TPCDigitizer.*", whose completion policy would process the sectors one by one
  return DataProcessorSpec{
    "tpc-digitizer-mt",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCDPLDigitizerMTTask>(nThreads, distortionType)},
    TPCDigitizerConditions::options()};
}

} // namespace tpc
} // namespace o2
