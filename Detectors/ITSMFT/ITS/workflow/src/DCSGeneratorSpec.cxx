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

#include "ITSWorkflow/DCSGeneratorSpec.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointGenerator.h"
#include "DetectorsDCS/DataPointCreator.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include <TDatime.h>
#include <random>
#include <variant>
#include <string>
#include <algorithm>

using namespace o2::its;

namespace
{
std::vector<o2::dcs::DataPointCompositeObject> generate(std::vector<std::string> aliases, int val)
{
  std::vector<o2::dcs::DataPointCompositeObject> dataPoints;

  for (auto alias : aliases) {
    dataPoints.emplace_back(o2::dcs::createDataPointCompositeObject(alias, val, 1, 0));
  }

  return dataPoints;
}

//______________________________________________________________________________________________________________
// class ITSDCSDataGenerator
class ITSDCSDataGenerator : public o2::framework::Task
{
 public:
  ITSDCSDataGenerator(o2::header::DataDescription description);

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void fillAliasesStrobeDuration();

 private:
  uint64_t mMaxTF = 1e6;
  uint64_t mTFs = 0;
  uint64_t mMaxCyclesNoFullMap;
  uint64_t changeAfterTF = 0;
  int valueA = 190;
  int valueB = 190;
  bool isStrobeDurationData = false;
  std::vector<std::string> mAliases;
  std::vector<o2::dcs::test::HintType> mDataPointHints;
  o2::header::DataDescription mDataDescription;
};

ITSDCSDataGenerator::ITSDCSDataGenerator(o2::header::DataDescription description) : mDataDescription(description) {}

void ITSDCSDataGenerator::fillAliasesStrobeDuration()
{

  // Aliases in this case are in the format: ITS_L0_00_STROBE
  // Here we fill them for every stave in mAliases
  int nStaves[] = {12, 16, 20, 24, 30, 42, 48};
  for (int iL = 0; iL < 7; iL++) {
    for (int iS = 0; iS < nStaves[iL]; iS++) {
      std::string stv = iS > 9 ? std::to_string(iS) : std::string(1, '0').append(std::to_string(iS));
      mAliases.push_back("ITS_L" + std::to_string(iL) + "_" + stv + "_STROBE");
    }
  }
}

void ITSDCSDataGenerator::init(o2::framework::InitContext& ic)
{
  mMaxTF = ic.options().get<int64_t>("max-timeframes");
  mMaxCyclesNoFullMap = ic.options().get<int64_t>("max-cycles-no-full-map");
  isStrobeDurationData = ic.options().get<bool>("generate-strobe-duration-data");
  changeAfterTF = ic.options().get<int64_t>("change-after-n-timeframes");
  valueA = ic.options().get<int>("value-a");
  valueB = ic.options().get<int>("value-b");

  if (isStrobeDurationData) {
    fillAliasesStrobeDuration();
  }
}

void ITSDCSDataGenerator::run(o2::framework::ProcessingContext& pc)
{
  auto input = pc.inputs().begin();
  uint64_t tfid = o2::header::get<o2::framework::DataProcessingHeader*>((*input).header)->startTime;
  if (tfid >= mMaxTF) {
    LOG(info) << "ITS DCS Data generator reached TF " << tfid << ", stopping";
    pc.services().get<o2::framework::ControlService>().endOfStream();
    pc.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
  }

  // generate data simulating ADAPOS
  bool doGen = mTFs % mMaxCyclesNoFullMap == 0;
  std::vector<o2::dcs::DataPointCompositeObject> dpcoms;
  if (doGen) {
    dpcoms = generate(mAliases, mTFs > changeAfterTF ? valueB : valueA);
  }

  LOG(info) << "TF " << tfid << " has generated " << dpcoms.size() << " DPs";
  auto& timingInfo = pc.services().get<o2::framework::TimingInfo>();
  auto timeNow = std::chrono::system_clock::now();
  timingInfo.creation = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms

  pc.outputs().snapshot(Output{"ITS", mDataDescription, 0}, dpcoms);
  mTFs++;
}
} // namespace

namespace o2::its
{
o2::framework::DataProcessorSpec getITSDCSDataGeneratorSpec(const char* detName)
{
  std::string desc{detName};
  desc += "DATAPOINTS";

  o2::header::DataDescription dd;

  dd.runtimeInit(desc.c_str(), desc.size());

  return DataProcessorSpec{
    "its-dcs-data-generator",
    Inputs{},
    Outputs{{{"outputDCS"}, "ITS", dd}},
    AlgorithmSpec{adaptFromTask<ITSDCSDataGenerator>(dd)},
    Options{
      {"change-after-n-timeframes", VariantType::Int64, 99999999999ll, {"change value generated after n timeframes: do not change val by default"}},
      {"value-a", VariantType::Int, 0, {"First value to be generated, will change to value-b after nTF = change-after-n-timeframes"}},
      {"value-b", VariantType::Int, 1, {"Second  value to be generated, will be after value-a once nTF = change-after-n-timeframes has been reached"}},
      {"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}},
      {"max-cycles-no-full-map", VariantType::Int64, 6000ll, {"max num of cycles between the sending of 2 full maps"}},
      {"generate-strobe-duration-data", VariantType::Bool, false, {"enable generation of DCS data containing the strobe duration in BCs"}}}};
}
} // namespace o2::its
