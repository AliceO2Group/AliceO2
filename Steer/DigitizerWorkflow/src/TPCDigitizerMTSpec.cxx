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
#include "Framework/CCDBParamSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsBase/Detector.h"
#include "SimConfig/DigiParams.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterGas.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSpaceCharge/SpaceCharge.h"
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
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace o2::framework;
using SC = o2::tpc::SpaceCharge<float>;

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
  void processSector(const steer::DigitizationContext& context, int sector, ULong64_t timeframe, bool withMCTruth, std::mutex& cdbMutex)
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
    digitizeSector(digitizer, context, chains, sector, timeframe, buffers, onFlush, &cdbMutex);
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
    mRecalcDistortions = !(ic.options().get<bool>("do-not-recalculate-distortions"));
    SC::setNThreads(ic.options().get<int>("n-threads-distortions"));
    mUseCalibrationsFromCCDB = ic.options().get<bool>("TPCuseCCDB");
    mMeanLumiDistortions = ic.options().get<float>("meanLumiDistortions");
    mMeanLumiDistortionsDerivative = ic.options().get<float>("meanLumiDistortionsDerivative");
    LOG(info) << "TPC calibrations from CCDB: " << mUseCalibrationsFromCCDB;

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
    }

    // the merger picks up every lane file in the directory, so files of an earlier run must not survive
    for (const auto& entry : std::filesystem::directory_iterator(".")) {
      const auto name = entry.path().filename().string();
      if (name.rfind("tpc_driftime_digits_lane", 0) == 0 && entry.path().extension() == ".root") {
        std::filesystem::remove(entry.path());
      }
    }

    ROOT::EnableThreadSafety();
  }

  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
  {
    if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
      return;
    }
    // the maps are set on the first digitizer; the others share them once they are final
    auto& digitizer = mWorkers[0]->digitizer;
    if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCDIST", 0)) {
      LOGP(info, "Updating distortion map");
      digitizer.setUseSCDistortions(static_cast<SC*>(obj));
      if (mMeanLumiDistortions >= 0) {
        digitizer.setMeanLumiDistortions(mMeanLumiDistortions);
      }
    }
    if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCDISTDERIV", 0)) {
      LOGP(info, "Updating reference distortion map");
      digitizer.setSCDistortionsDerivative(static_cast<SC*>(obj));
      if (mMeanLumiDistortionsDerivative >= 0) {
        digitizer.setMeanLumiDistortionsDerivative(mMeanLumiDistortionsDerivative);
      }
    }
    if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCGASPARAM", 0)) {
      LOGP(info, "TPC gas param updated");
      ParameterGas::Instance().printKeyValues(true, true);
    }
    if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCGEMPARAM", 0)) {
      LOGP(info, "TPC GEM param updated");
      ParameterGEM::Instance().printKeyValues(true, true);
    }
  }

  void run(framework::ProcessingContext& pc)
  {
    LOG(info) << "Processing multithreaded TPC digitization";

    auto& cdb = o2::tpc::CDBInterface::instance();
    cdb.setUseDefaults(!mUseCalibrationsFromCCDB);
    // whatever are global settings for CCDB usage, we have to extract the TPC vdrift from CCDB for anchored simulations
    mTPCVDriftHelper.extractCCDBInputs(pc);
    pc.inputs().get<ParameterGas*>("gasparam");
    pc.inputs().get<ParameterGEM*>("gemparam");
    auto& primary = mWorkers[0]->digitizer;
    if (mDistortionType) {
      pc.inputs().get<SC*>("tpcdistortions");
      if (mDistortionType == 2) {
        pc.inputs().get<SC*>("tpcdistortionsderiv");
        primary.setLumiScaleFactor();
        if (mRecalcDistortions) {
          // done once, with the threads of SpaceCharge, for all workers
          primary.recalculateDistortions();
        }
      }
      for (size_t i = 1; i < mWorkers.size(); ++i) {
        mWorkers[i]->digitizer.setSCDistortionsFrom(primary);
      }
    }

    if (mTPCVDriftHelper.isUpdated()) {
      const auto& vd = mTPCVDriftHelper.getVDriftObject();
      LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
           vd.corrFact, vd.refVDrift, vd.timeOffsetCorr, vd.refTimeOffset, mTPCVDriftHelper.getSourceName());
      for (auto& worker : mWorkers) {
        worker->digitizer.setVDrift(vd.getVDrift());
        worker->digitizer.setTDriftOffset(vd.getTimeOffset());
      }
      mTPCVDriftHelper.acknowledgeUpdate();
    }

    if (std::filesystem::exists("ThresholdMap.root")) {
      LOG(info) << "TPC: Using zero suppression map from 'ThresholdMap.root'";
      cdb.setThresholdMapFromFile("ThresholdMap.root");
    }

    if (std::filesystem::exists("GainMap.root")) {
      LOG(info) << "TPC: Using gain map from 'GainMap.root'";
      cdb.setGainMapFromFile("GainMap.root");
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
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>(*contextRef);
    const ULong64_t timeframe = DataRefUtils::getHeader<o2::header::DataHeader*>(*contextRef)->tfCounter;
    LOG(info) << "TPC: Processing " << sectors.size() << " sectors with " << context->getEventRecords().size() << " collisions";

    TStopwatch timer;
    timer.Start();

    const size_t nWorkers = std::min(mWorkers.size(), sectors.size());
    for (size_t i = 0; i < nWorkers; ++i) {
      context->initSimChains(o2::detectors::DetID::TPC, mWorkers[i]->chains);
    }

    std::atomic<size_t> nextSector{0};
    // serialises CDBInterface access; the CCDB timestamp must not change while the workers run
    std::mutex cdbMutex;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < nWorkers; ++i) {
      threads.emplace_back([&, worker = mWorkers[i].get()]() {
        try {
          for (size_t is = nextSector++; is < sectors.size(); is = nextSector++) {
            LOG(info) << "TPC: Worker " << worker->id << " processing sector " << sectors[is];
            worker->processSector(*context, sectors[is], timeframe, mWithMCTruth, cdbMutex);
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
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  int mNThreads = 1;
  int mDistortionType = 0;
  bool mWriteGRP = true;
  bool mWithMCTruth = true;
  bool mUseCalibrationsFromCCDB = false;
  bool mRecalcDistortions = false;
  float mMeanLumiDistortions = -1;
  float mMeanLumiDistortionsDerivative = -1;
};

o2::framework::DataProcessorSpec getTPCDigitizerMTSpec(int nThreads, std::vector<int> const& sectors, int distortionType)
{
  std::vector<InputSpec> inputs;
  for (auto sector : sectors) {
    inputs.emplace_back("collisioncontext" + std::to_string(sector), "SIM", "COLLISIONCONTEXT", static_cast<o2::header::DataHeader::SubSpecificationType>(sector), Lifetime::Timeframe);
  }
  o2::tpc::VDriftHelper::requestCCDBInputs(inputs);
  inputs.emplace_back("gasparam", o2::header::gDataOriginTPC, "TPCGASPARAM", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::ParGas)));
  inputs.emplace_back("gemparam", o2::header::gDataOriginTPC, "TPCGEMPARAM", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::ParGEM)));
  if (distortionType) {
    inputs.emplace_back("tpcdistortions", o2::header::gDataOriginTPC, "TPCDIST", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::DistortionMapMC), {}, 1)); // time-dependent
    if (distortionType == 2) {
      inputs.emplace_back("tpcdistortionsderiv", o2::header::gDataOriginTPC, "TPCDISTDERIV", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::DistortionMapDerivMC), {}, 1)); // time-dependent
    }
  }
  std::vector<OutputSpec> outputs{{"TPC", "ROMode", 0, Lifetime::Timeframe}};

  // the name must not match "TPCDigitizer.*", whose completion policy would process the sectors one by one
  return DataProcessorSpec{
    "tpc-digitizer-mt",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCDPLDigitizerMTTask>(nThreads, distortionType)},
    Options{
      {"TPCtriggered", VariantType::Bool, false, {"Impose triggered RO mode (default: continuous)"}},
      {"TPCuseCCDB", VariantType::Bool, false, {"true: load calibrations from CCDB; false: use random calibratoins"}},
      {"meanLumiDistortions", VariantType::Float, -1.f, {"override lumi of distortion object if >=0"}},
      {"meanLumiDistortionsDerivative", VariantType::Float, -1.f, {"override lumi of derivative distortion object if >=0"}},
      {"do-not-recalculate-distortions", VariantType::Bool, false, {"Do not recalculate the distortions"}},
      {"n-threads-distortions", VariantType::Int, 4, {"Number of threads used for the calculation of the distortions"}},
    }};
}

} // namespace tpc
} // namespace o2
