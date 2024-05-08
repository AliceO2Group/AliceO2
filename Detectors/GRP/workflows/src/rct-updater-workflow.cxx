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

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "Framework/Task.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCalibration/Utils.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{{"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

#include "DetectorsBase/GRPGeomHelper.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsParameters/GRPECSObject.h"

namespace o2::rct
{
class RCTUpdaterSpec : public o2::framework::Task
{
 public:
  RCTUpdaterSpec(std::shared_ptr<o2::base::GRPGeomRequest> gr) : mGGCCDBRequest(gr) {}
  ~RCTUpdaterSpec() final = default;

  void init(InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
    mUpdateInterval = std::max(0.1f, ic.options().get<float>("update-interval"));
    auto ccdb = ic.options().get<std::string>("ccdb-server");
    if (!ccdb.empty() && ccdb != "none") {
      mCCDBApi = std::make_unique<o2::ccdb::CcdbApi>();
      mCCDBApi->init(ic.options().get<std::string>("ccdb-server"));
    } else {
      LOGP(warn, "No ccdb server provided, no RCT update will be done");
    }
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    auto tinfo = pc.services().get<o2::framework::TimingInfo>();
    if (tinfo.globalRunNumberChanged) { // do we have RCT object?
      const auto* grp = o2::base::GRPGeomHelper::instance().getGRPECS();
      mNHBFPerTF = grp->getNHBFPerTF();
      if (mNHBFPerTF < 1) {
        mNHBFPerTF = 32;
      }
      mRunNumber = tinfo.runNumber;
      mUpdateIntervalTF = uint32_t(mUpdateInterval / (mNHBFPerTF * o2::constants::lhc::LHCOrbitMUS * 1e-6)); // convert update interval in seconds to interval in TFs
      LOGP(info, "Will update RCT after {} TFs of {} HBFs ({}s was requested)", mUpdateIntervalTF, mNHBFPerTF, mUpdateInterval);
      mOrbitReset = o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS();
      mMinOrbit = 0xffffffff;
      mMaxOrbit = 0;
      if (grp->getRunType() == o2::parameters::GRPECS::PHYSICS || grp->getRunType() == o2::parameters::GRPECS::COSMICS) {
        mEnabled = true;
      } else {
        LOGP(warning, "Run {} type is {}, disabling RCT update", mRunNumber, o2::parameters::GRPECS::RunTypeNames[grp->getRunType()]);
        mEnabled = false;
      }
      if (mEnabled) {
        if (mCCDBApi) {
          auto md = mCCDBApi->retrieveHeaders("RCT/Info/RunInformation", {}, grp->getRun());
          if (md.empty()) {
            mEnabled = false;
            LOGP(alarm, "RCT object is missing for {} run {}, disabling RCT updater", o2::parameters::GRPECS::RunTypeNames[grp->getRunType()], grp->getRun());
          }
        }
      }
    }
    if (mEnabled) {
      if (tinfo.firstTForbit < mMinOrbit) {
        mMinOrbit = tinfo.firstTForbit;
      }
      if (tinfo.firstTForbit > mMaxOrbit) {
        mMaxOrbit = tinfo.firstTForbit;
      }
      if (tinfo.tfCounter > mLastTFUpdate + mUpdateIntervalTF) { // need to update
        mLastTFUpdate = tinfo.tfCounter;
        updateRCT();
      }
    }
  }

  void endOfStream(framework::EndOfStreamContext& ec) final
  {
    if (mEnabled) {
      updateRCT();
      mEnabled = false;
    }
  }

  void stop() final
  {
    if (mEnabled) {
      updateRCT();
      mEnabled = false;
    }
  }

  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      return;
    }
  }

  void updateRCT()
  {
    std::map<std::string, std::string> mdRCT;
    if (mMinOrbit > mMaxOrbit) {
      return;
    }
    mdRCT["STF"] = std::to_string(long(mMinOrbit * o2::constants::lhc::LHCOrbitMUS * 1e-3) + mOrbitReset);
    mdRCT["ETF"] = std::to_string(long((mMaxOrbit + mNHBFPerTF - 1) * o2::constants::lhc::LHCOrbitMUS * 1e-3) + mOrbitReset);
    long startValRCT = (long)mRunNumber;
    long endValRCT = (long)(mRunNumber + 1);
    if (mCCDBApi) {
      int retValRCT = mCCDBApi->updateMetadata("RCT/Info/RunInformation", mdRCT, startValRCT);
      if (retValRCT == 0) {
        LOGP(info, "Updated {}/RCT/Info/RunInformation object for run {} with TF start:{} end:{}", mCCDBApi->getURL(), mRunNumber, mdRCT["STF"], mdRCT["ETF"]);
      } else {
        LOGP(alarm, "Update of RCT object for run {} with TF start:{} end:{} FAILED, returned with code {}", mRunNumber, mdRCT["STF"], mdRCT["ETF"], retValRCT);
      }
    } else {
      LOGP(info, "CCDB update disabled, TF timestamp range is {}:{}", mdRCT["STF"], mdRCT["ETF"]);
    }
  }

 private:
  bool mEnabled = true;
  float mUpdateInterval = 1.;
  int mUpdateIntervalTF = 1;
  uint32_t mMinOrbit = 0xffffffff;
  uint32_t mMaxOrbit = 0;
  uint32_t mLastTFUpdate = 0;
  long mOrbitReset = 0;
  int mRunNumber = 0;
  int mNHBFPerTF = 32;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<o2::ccdb::CcdbApi> mCCDBApi;
};
} // namespace o2::rct

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"
#include "Framework/DataProcessorSpec.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  std::vector<InputSpec> inputs{{"ctfdone", "CTF", "DONE", 0, Lifetime::Timeframe}};
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "RCTUPD_DUMMY"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "RCTUPD_DUMMY"}, Lifetime::Sporadic);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                              true,                           // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              inputs,
                                                              true); // query only once all objects except mag.field
  specs.push_back(DataProcessorSpec{
    "rct-updater",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::rct::RCTUpdaterSpec>(ggRequest)},
    Options{
      {"update-interval", VariantType::Float, 1.f, {"update every ... seconds"}},
      {"ccdb-server", VariantType::String, "http://ccdb-test.cern.ch:8080", {"CCDB to update"}}}});
  return specs;
}
