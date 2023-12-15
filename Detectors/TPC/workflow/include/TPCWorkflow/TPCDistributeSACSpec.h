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

/// \file TPCDistributeSACspec.h
/// \brief TPC distribution of SACs for factorization
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jul 6, 2022

#ifndef O2_TPCDISTRIBUTESACSPEC_H
#define O2_TPCDISTRIBUTESACSPEC_H

#include <vector>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCCalibration/SACDecoder.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCDistributeSACSpec : public o2::framework::Task
{
 public:
  TPCDistributeSACSpec(const unsigned int timeframes, const unsigned int outlanes)
    : mTimeFrames{timeframes}, mOutLanes{outlanes}
  {
    // pre calculate data description for output
    mDataDescrOut.reserve(mOutLanes);
    for (int i = 0; i < mOutLanes; ++i) {
      mDataDescrOut.emplace_back(getDataDescriptionSACVec(i));
    }
  };

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto& sacPoint = pc.inputs().get<std::vector<o2::tpc::sac::DataPoint>>("sac");
    for (const auto& val : sacPoint) {
      for (int stack = 0; stack < o2::tpc::GEMSTACKS; ++stack) {
        const auto sac = val.currents[stack];
        mSACs[stack].emplace_back(sac);
      }
    }

    if (mCCDBTimeStamp == 0 && !sacPoint.empty()) {
      const auto reftime = pc.inputs().get<double>("reftime");
      mCCDBTimeStamp = static_cast<uint64_t>(reftime + sacPoint.front().time * o2::tpc::sac::Decoder::SampleDistanceTimeMS);
    }

    ++mProcessedTFs;
    if (mProcessedTFs == mTimeFrames) {
      sendOutput(pc);
      mProcessedTFs = 0;
      mCurrentOutLane = ++mCurrentOutLane % mOutLanes;
      mCCDBTimeStamp = 0;
      for (auto& sac : mSACs) {
        sac.clear();
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionSACCCDB() { return header::DataDescription{"SACCCDB"}; }

  /// return data description for aggregated SACs for given lane
  static header::DataDescription getDataDescriptionSACVec(const int lane)
  {
    const std::string name = fmt::format("SACVEC{}", lane).data();
    header::DataDescription description;
    description.runtimeInit(name.substr(0, 16).c_str());
    return description;
  }

 private:
  const unsigned int mTimeFrames{};                             ///< number of TFs per aggregation interval
  const unsigned int mOutLanes{};                               ///< number of output lanes
  int mProcessedTFs{0};                                         ///< number of processed time frames to keep track of when the writing to CCDB will be done
  std::array<std::vector<int32_t>, o2::tpc::GEMSTACKS> mSACs{}; ///< vector containing the sacs
  unsigned int mCurrentOutLane{0};                              ///< index for keeping track of the current output lane
  uint64_t mCCDBTimeStamp{0};                                   ///< time stamp of first SACs which are received for the current aggreagtion interval, which is used for setting the time when writing to the CCDB
  std::vector<header::DataDescription> mDataDescrOut{};         ///< data description for the different output lanes

  void sendOutput(o2::framework::ProcessingContext& pc)
  {
    LOGP(info, "Sending SACs on lane: {} for {} TFs", mCurrentOutLane, mProcessedTFs);
    pc.outputs().snapshot(Output{gDataOriginTPC, getDataDescriptionSACCCDB(), 0}, mCCDBTimeStamp);
    for (unsigned int i = 0; i < o2::tpc::GEMSTACKS; ++i) {
      pc.outputs().snapshot(Output{gDataOriginTPC, mDataDescrOut[mCurrentOutLane], header::DataHeader::SubSpecificationType{i}}, mSACs[i]);
    }
  }
};

DataProcessorSpec getTPCDistributeSACSpec(const unsigned int timeframes, const unsigned int outlanes)
{
  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back(InputSpec{"sac", gDataOriginTPC, "DECODEDSAC", 0, Lifetime::Sporadic});
  inputSpecs.emplace_back(InputSpec{"reftime", gDataOriginTPC, "REFTIMESAC", 0, Lifetime::Sporadic});

  std::vector<OutputSpec> outputSpecs;
  for (unsigned int lane = 0; lane < outlanes; ++lane) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCDistributeSACSpec::getDataDescriptionSACVec(lane)}, Lifetime::Sporadic);
  }
  outputSpecs.emplace_back(gDataOriginTPC, TPCDistributeSACSpec::getDataDescriptionSACCCDB(), 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-distribute-sac",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCDistributeSACSpec>(timeframes, outlanes)}};
}

} // namespace o2::tpc

#endif
