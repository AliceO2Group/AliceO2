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

#include <gsl/span>

#include "TRDWorkflow/TRDTrackletTransformerSpec.h"

#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "CommonDataFormat/IRFrame.h"
#include "Framework/CCDBParamSpec.h"

using namespace o2::framework;
using namespace o2::globaltracking;

namespace o2
{
namespace trd
{

void TRDTrackletTransformerSpec::init(o2::framework::InitContext& ic)
{
  mTransformer.init();
}

void TRDTrackletTransformerSpec::run(o2::framework::ProcessingContext& pc)
{
  LOG(info) << "Running tracklet transformer";

  o2::globaltracking::RecoContainer inputData;
  inputData.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions

  auto tracklets = pc.inputs().get<gsl::span<Tracklet64>>("trdtracklets");
  auto trigRecs = pc.inputs().get<gsl::span<TriggerRecord>>("trdtriggerrec");

  std::vector<CalibratedTracklet> calibratedTracklets(tracklets.size());

  std::vector<char> trigRecBitfield(trigRecs.size()); // flag TRD IR with ITS data (std::vector<bool> platform dependend)
  int nTrackletsTransformed = 0;

  if (mTrigRecFilterActive) {
    const auto irFrames = inputData.getIRFramesITS();
    size_t lastMatchedIdx = 0; // ITS IR are sorted in time and do not overlap
    for (const auto& irFrame : irFrames) {
      if (!irFrame.info) { // skip IRFrames where ITS did not find any track
        continue;
      }
      for (auto j = lastMatchedIdx; j < trigRecs.size(); ++j) {
        const auto& trigRec = trigRecs[j];
        if (trigRec.getBCData() >= irFrame.getMin()) {
          if (trigRec.getBCData() <= irFrame.getMax()) {
            // TRD interaction record inside ITS frame
            trigRecBitfield[j] = 1;
            lastMatchedIdx = j;
          } else {
            // too late, also the higher trigger records won't match
            break;
          }
        }
      }
      LOGF(debug, "ITS IR Frame start: %li, end: %li", irFrame.getMin().toLong(), irFrame.getMax().toLong());
    }
    /*
    // for debugging: print TRD trigger times which are accepted and which are filtered out
    for (int j = 0; j < trigRecs.size(); ++j) {
      const auto& trigRec = trigRecs[j];
      if (!trigRecBitfield[j]) {
        LOGF(debug, "Could not find ITS info for TRD trigger %i: %li", j, trigRec.getBCData().toLong());
      } else {
        LOGF(debug, "Found ITS info for TRD trigger %i: %li", j, trigRec.getBCData().toLong());
      }
    }
    */
  } else {
    // fill bitmask with 1
    std::fill(trigRecBitfield.begin(), trigRecBitfield.end(), 1);
  }

  if (mTrigRecFilterActive) {
    // skip tracklets from TRD triggers without ITS data
    for (size_t iTrig = 0; iTrig < trigRecs.size(); ++iTrig) {
      if (!trigRecBitfield[iTrig]) {
        continue;
      } else {
        const auto& trigRec = trigRecs[iTrig];
        for (int iTrklt = trigRec.getFirstTracklet(); iTrklt < trigRec.getFirstTracklet() + trigRec.getNumberOfTracklets(); ++iTrklt) {
          calibratedTracklets[iTrklt] = mTransformer.transformTracklet(tracklets[iTrklt]);
          ++nTrackletsTransformed;
        }
      }
    }
  } else {
    // transform all tracklets
    for (size_t iTrklt = 0; iTrklt < tracklets.size(); ++iTrklt) {
      calibratedTracklets[iTrklt] = mTransformer.transformTracklet(tracklets[iTrklt]);
      ++nTrackletsTransformed;
    }
  }

  LOGF(info, "Found %lu tracklets. Applied filter for ITS IR frames: %i. Transformed %i tracklets.", tracklets.size(), mTrigRecFilterActive, nTrackletsTransformed);

  pc.outputs().snapshot(Output{"TRD", "CTRACKLETS", 0, Lifetime::Timeframe}, calibratedTracklets);
  pc.outputs().snapshot(Output{"TRD", "TRIGRECMASK", 0, Lifetime::Timeframe}, trigRecBitfield);
}

void TRDTrackletTransformerSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  pc.inputs().get<o2::trd::CalVdriftExB*>("calvdexb"); // just to trigger the finaliseCCDB
}

void TRDTrackletTransformerSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("TRD", "CALVDRIFTEXB", 0)) {
    LOG(info) << "CalVdriftExB object has been updated";
    mTransformer.setCalVdriftExB((const o2::trd::CalVdriftExB*)obj);
  }
}

o2::framework::DataProcessorSpec getTRDTrackletTransformerSpec(bool trigRecFilterActive)
{
  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  if (trigRecFilterActive) {
    dataRequest->requestIRFramesITS();
  }
  auto& inputs = dataRequest->inputs;
  inputs.emplace_back("trdtracklets", "TRD", "TRACKLETS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trdtriggerrec", "TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
  inputs.emplace_back("calvdexb", "TRD", "CALVDRIFTEXB", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/CalVdriftExB"));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "CTRACKLETS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRIGRECMASK", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TRDTRACKLETTRANSFORMER",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDTrackletTransformerSpec>(dataRequest, trigRecFilterActive)},
    Options{}};
}

} //end namespace trd
} //end namespace o2
