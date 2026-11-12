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

#include "TPCSectorDigitization.h"
#include <sstream>
#include <string>
#include <TChain.h>
#include "DetectorsRaw/HBFUtils.h"
#include "Framework/Logger.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "TPCBase/Sector.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCBaseRecSim/CDBTypes.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InitContext.h"
#include "Framework/InputRecord.h"
#include "Framework/ProcessingContext.h"
#include <filesystem>

namespace o2::tpc
{

namespace
{
std::string getBranchNameLeft(int sector)
{
  std::stringstream branchnamestreamleft;
  branchnamestreamleft << "TPCHitsShiftedSector" << int(o2::tpc::Sector::getLeft(o2::tpc::Sector(sector)));
  return branchnamestreamleft.str();
}

std::string getBranchNameRight(int sector)
{
  std::stringstream branchnamestreamright;
  branchnamestreamright << "TPCHitsShiftedSector" << sector;
  return branchnamestreamright.str();
}
} // namespace

std::vector<dataformats::RangeReference<int, int>> digitizeSector(Digitizer& digitizer, const steer::DigitizationContext& context,
                                                                  const std::vector<TChain*>& simChains, int sector, ULong64_t timeframe,
                                                                  SectorFlushBuffers& buffers, const std::function<void()>& onFlush)
{
  std::vector<dataformats::RangeReference<int, int>> events;
  const bool isContinuous = digitizer.isContinuousReadout();
  const auto& irecords = context.getEventRecords();
  const auto& eventParts = context.getEventParts();

  digitizer.setSector(sector);
  digitizer.setSeedingTimeframe(timeframe);
  digitizer.init();

  size_t digitCounter = 0;
  auto flush = [&](bool finalFlush = false) {
    buffers.digits.clear();
    buffers.labels.clear();
    buffers.commonMode.clear();
    digitizer.flush(buffers.digits, buffers.labels, buffers.commonMode, finalFlush);
    LOG(info) << "TPC: Flushed " << buffers.digits.size() << " digits, " << buffers.labels.getNElements() << " labels and " << buffers.commonMode.size() << " common mode entries";
    onFlush();
    digitCounter += buffers.digits.size();
  };

  if (isContinuous) {
    auto& hbfu = o2::raw::HBFUtils::Instance();
    double time = hbfu.getFirstIRofTF(o2::InteractionRecord(0, hbfu.orbitFirstSampled)).bc2ns() / 1000.;
    digitizer.setOutputDigitTimeOffset(time);
    if (!irecords.empty()) {
      digitizer.setStartTime(irecords[0].getTimeNS() / 1000.f);
    }
  }

  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  for (int collID = 0; collID < irecords.size(); ++collID) {
    const double eventTime = irecords[collID].getTimeNS() / 1000.f;
    LOG(info) << "TPC: Event time " << eventTime << " us";
    digitizer.setEventTime(eventTime);
    if (!isContinuous) {
      digitizer.setStartTime(eventTime);
    }
    size_t startSize = digitCounter;

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    for (auto& part : eventParts[collID]) {
      const int eventID = part.entryID;
      const int sourceID = part.sourceID;

      // get the hits for this event and this source
      std::vector<o2::tpc::HitGroup> hitsLeft;
      std::vector<o2::tpc::HitGroup> hitsRight;
      context.retrieveHits(simChains, getBranchNameLeft(sector).c_str(), part.sourceID, part.entryID, &hitsLeft);
      context.retrieveHits(simChains, getBranchNameRight(sector).c_str(), part.sourceID, part.entryID, &hitsRight);
      LOG(debug) << "TPC: Found " << hitsLeft.size() << " hit groups left and " << hitsRight.size() << " hit groups right in collision " << collID << " eventID " << part.entryID;

      digitizer.process(hitsLeft, eventID, sourceID);
      digitizer.process(hitsRight, eventID, sourceID);

      flush();

      if (!isContinuous) {
        events.emplace_back(startSize, buffers.digits.size());
      }
    }
  }

  // final flushing step; getting everything not yet written out
  if (isContinuous) {
    LOG(info) << "TPC: Final flush";
    flush(true);
    events.emplace_back(0, digitCounter); // all digits are grouped to 1 super-event pseudo-triggered mode
  }
  return events;
}

std::vector<framework::ConfigParamSpec> TPCDigitizerConditions::options()
{
  using framework::VariantType;
  return {
    {"TPCtriggered", VariantType::Bool, false, {"Impose triggered RO mode (default: continuous)"}},
    {"TPCuseCCDB", VariantType::Bool, false, {"true: load calibrations from CCDB; false: use random calibratoins"}},
    {"meanLumiDistortions", VariantType::Float, -1.f, {"override lumi of distortion object if >=0"}},
    {"meanLumiDistortionsDerivative", VariantType::Float, -1.f, {"override lumi of derivative distortion object if >=0"}},
    {"do-not-recalculate-distortions", VariantType::Bool, false, {"Do not recalculate the distortions"}},
    {"n-threads-distortions", VariantType::Int, 4, {"Number of threads used for the calculation of the distortions"}},
  };
}

void TPCDigitizerConditions::addInputs(std::vector<framework::InputSpec>& inputs, int distortionType)
{
  using framework::Lifetime;
  VDriftHelper::requestCCDBInputs(inputs);
  inputs.emplace_back("gasparam", o2::header::gDataOriginTPC, "TPCGASPARAM", 0, Lifetime::Condition, framework::ccdbParamSpec(CDBTypeMap.at(CDBType::ParGas)));
  inputs.emplace_back("gemparam", o2::header::gDataOriginTPC, "TPCGEMPARAM", 0, Lifetime::Condition, framework::ccdbParamSpec(CDBTypeMap.at(CDBType::ParGEM)));
  if (distortionType) {
    inputs.emplace_back("tpcdistortions", o2::header::gDataOriginTPC, "TPCDIST", 0, Lifetime::Condition, framework::ccdbParamSpec(CDBTypeMap.at(CDBType::DistortionMapMC), {}, 1)); // time-dependent
    // load derivative map in case scaling was requested
    if (distortionType == 2) {
      inputs.emplace_back("tpcdistortionsderiv", o2::header::gDataOriginTPC, "TPCDISTDERIV", 0, Lifetime::Condition, framework::ccdbParamSpec(CDBTypeMap.at(CDBType::DistortionMapDerivMC), {}, 1)); // time-dependent
    }
  }
}

void TPCDigitizerConditions::init(framework::InitContext& ic, int distortionType)
{
  mDistortionType = distortionType;
  mRecalcDistortions = !(ic.options().get<bool>("do-not-recalculate-distortions"));
  SpaceCharge<float>::setNThreads(ic.options().get<int>("n-threads-distortions"));
  mUseCalibrationsFromCCDB = ic.options().get<bool>("TPCuseCCDB");
  mMeanLumiDistortions = ic.options().get<float>("meanLumiDistortions");
  mMeanLumiDistortionsDerivative = ic.options().get<float>("meanLumiDistortionsDerivative");
  LOG(info) << "TPC calibrations from CCDB: " << mUseCalibrationsFromCCDB;
}

bool TPCDigitizerConditions::finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj, Digitizer& digitizer)
{
  using framework::ConcreteDataMatcher;
  using SC = SpaceCharge<float>;
  if (mVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return true;
  }
  if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCDIST", 0)) {
    LOGP(info, "Updating distortion map");
    digitizer.setUseSCDistortions(static_cast<SC*>(obj));
    if (mMeanLumiDistortions >= 0) {
      digitizer.setMeanLumiDistortions(mMeanLumiDistortions);
    }
    return true;
  }
  if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCDISTDERIV", 0)) {
    LOGP(info, "Updating reference distortion map");
    digitizer.setSCDistortionsDerivative(static_cast<SC*>(obj));
    if (mMeanLumiDistortionsDerivative >= 0) {
      digitizer.setMeanLumiDistortionsDerivative(mMeanLumiDistortionsDerivative);
    }
    return true;
  }
  if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCGASPARAM", 0)) {
    LOGP(info, "TPC gas param updated");
    ParameterGas::Instance().printKeyValues(true, true);
    return true;
  }
  if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCGEMPARAM", 0)) {
    LOGP(info, "TPC GEM param updated");
    ParameterGEM::Instance().printKeyValues(true, true);
    return true;
  }
  return false;
}

void TPCDigitizerConditions::update(framework::ProcessingContext& pc, Digitizer& digitizer, const std::function<void(float, float)>& setVDrift)
{
  using SC = SpaceCharge<float>;
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults(!mUseCalibrationsFromCCDB);
  // whatever are global settings for CCDB usage, we have to extract the TPC vdrift from CCDB for anchored simulations
  mVDriftHelper.extractCCDBInputs(pc);
  pc.inputs().get<ParameterGas*>("gasparam");
  pc.inputs().get<ParameterGEM*>("gemparam");
  if (mDistortionType) {
    pc.inputs().get<SC*>("tpcdistortions");
    if (mDistortionType == 2) {
      pc.inputs().get<SC*>("tpcdistortionsderiv");
      digitizer.setLumiScaleFactor();
      if (mRecalcDistortions) {
        digitizer.recalculateDistortions();
      }
    }
  }

  if (mVDriftHelper.isUpdated()) {
    const auto& vd = mVDriftHelper.getVDriftObject();
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         vd.corrFact, vd.refVDrift, vd.timeOffsetCorr, vd.refTimeOffset, mVDriftHelper.getSourceName());
    setVDrift(vd.getVDrift(), vd.getTimeOffset());
    mVDriftHelper.acknowledgeUpdate();
  }

  if (std::filesystem::exists("ThresholdMap.root")) {
    LOG(info) << "TPC: Using zero suppression map from 'ThresholdMap.root'";
    cdb.setThresholdMapFromFile("ThresholdMap.root");
  }

  if (std::filesystem::exists("GainMap.root")) {
    LOG(info) << "TPC: Using gain map from 'GainMap.root'";
    cdb.setGainMapFromFile("GainMap.root");
  }
}

} // namespace o2::tpc
