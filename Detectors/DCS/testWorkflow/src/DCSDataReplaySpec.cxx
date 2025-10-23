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

#include "DCStestWorkflow/DCSDataReplaySpec.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointGenerator.h"
#include "DetectorsDCS/DataPointCreator.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "TTree.h"
#include <cmath>
#include <variant>
#include <string>
#include <algorithm>
#include <vector>

using namespace o2::framework;

namespace
{

class DCSDataReplayer : public o2::framework::Task
{
 public:
  DCSDataReplayer(std::vector<o2::dcs::test::HintType> hints, o2::header::DataDescription description);

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

 private:
  std::string mInputFileName{};
  double mTime{};
  double mValue{};
  char mAlias[50];
  uint64_t mMaxTF;
  uint64_t mTFs = 0;
  int deltaTimeSendData = -1;
  uint32_t startTime = -1;
  uint32_t endTime = 0;
  std::vector<std::vector<int>> dataIndicesPerTF;
  TTree mInputData;
  std::vector<o2::dcs::test::HintType> mDataPointHints;
  o2::header::DataDescription mDataDescription;
};

DCSDataReplayer::DCSDataReplayer(std::vector<o2::dcs::test::HintType> hints,
                                 o2::header::DataDescription description) : mDataPointHints(hints),
                                                                            mDataDescription(description) {}

void DCSDataReplayer::init(o2::framework::InitContext& ic)
{
  mMaxTF = ic.options().get<int64_t>("max-timeframes");
  mInputFileName = ic.options().get<std::string>("input-file");
  deltaTimeSendData = ic.options().get<int>("delta-time-send-data");
  mInputData.ReadFile(mInputFileName.data(), "time/D:alias/C:value/D", ';');
  mInputData.SetBranchAddress("time", &mTime);
  mInputData.SetBranchAddress("value", &mValue);
  mInputData.SetBranchAddress("alias", mAlias);
}

void DCSDataReplayer::run(o2::framework::ProcessingContext& pc)
{
  auto input = pc.inputs().begin();
  uint64_t tfid = o2::header::get<o2::framework::DataProcessingHeader*>((*input).header)->startTime;
  if (tfid >= mMaxTF) {
    LOG(info) << "Data generator reached TF " << tfid << ", stopping";
    pc.services().get<o2::framework::ControlService>().endOfStream();
    pc.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
    return;
  }

  std::vector<o2::dcs::DataPointCompositeObject> dpcoms;

  for (Long64_t iEntry = 0; iEntry < mInputData.GetEntries(); ++iEntry) {
    int entryTree = iEntry;

    // load only releavant entries if requested
    if (deltaTimeSendData > 0 && tfid > 2) {

      if (tfid - 1 >= dataIndicesPerTF.size()) {
        LOGP(warning, "TF ID {} is larger than the number of TFs in dataIndicesPerTF: {}", tfid, dataIndicesPerTF.size());
        break;
      }

      if (iEntry >= dataIndicesPerTF[tfid - 1].size()) {
        break;
      } else {
        entryTree = dataIndicesPerTF[tfid - 1][iEntry];
      }
    }

    mInputData.GetEntry(entryTree);
    const auto ultime = uint64_t(std::round(mTime * 1000));
    const auto seconds = uint32_t(ultime / 1000);
    const auto msec = uint16_t(ultime % 1000);
    if (deltaTimeSendData > 0) {
      // send data in packages
      if (tfid == 0) {
        startTime = std::min(startTime, seconds);
        endTime = std::max(endTime, seconds);
        if (iEntry == mInputData.GetEntries() - 1) {
          const int totalTFs = (endTime - startTime) / deltaTimeSendData + 1;
          dataIndicesPerTF.resize(totalTFs);
          LOGP(info, "Sending data from {} to {} with {} TFs", startTime, endTime, totalTFs);
        }
      } else {
        if (tfid == 1) {
          const int index = (seconds - startTime) / deltaTimeSendData;
          dataIndicesPerTF[index].emplace_back(iEntry);
        }
        const uint64_t startTimeTF = startTime + (tfid - 1) * deltaTimeSendData;
        const uint64_t endTimeTF = startTimeTF + deltaTimeSendData;
        if (seconds >= startTimeTF && seconds < endTimeTF) {
          dpcoms.emplace_back(o2::dcs::createDataPointCompositeObject(mAlias, float(mValue), seconds, msec));
          // check if all data has been processed
          if (seconds == endTime) {
            mMaxTF = tfid;
          }
        }
      }
    } else {
      dpcoms.emplace_back(o2::dcs::createDataPointCompositeObject(mAlias, float(mValue), seconds, msec));
    }
  }
  // auto dpcoms = generate(mDataPointHints, fraction, tfid);

  LOG(info)
    << "***************** TF " << tfid << " has generated " << dpcoms.size() << " DPs";
  pc.outputs().snapshot(Output{"DCS", mDataDescription, 0}, dpcoms);
  mTFs++;
}
} // namespace

namespace o2::dcs::test
{
o2::framework::DataProcessorSpec getDCSDataReplaySpec(std::vector<o2::dcs::test::HintType> hints,
                                                      const char* detName)
{
  std::string desc{detName};
  desc += "DATAPOINTS";

  o2::header::DataDescription dd;

  dd.runtimeInit(desc.c_str(), desc.size());

  return DataProcessorSpec{
    "dcs-random-data-generator",
    Inputs{},
    Outputs{{{"outputDCS"}, "DCS", dd}},
    AlgorithmSpec{adaptFromTask<DCSDataReplayer>(hints, dd)},
    Options{
      {"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}},
      {"delta-fraction", VariantType::Float, 0.05f, {"fraction of data points to put in the delta"}},
      {"delta-time-send-data", VariantType::Int, -1, {"if larger than zero the data will be send in time intervals of this size"}},
      {"input-file", VariantType::String, "", {"Input file with data to play back"}}}};
}
} // namespace o2::dcs::test
