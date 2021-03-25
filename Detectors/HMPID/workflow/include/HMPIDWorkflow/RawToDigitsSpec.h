// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_SRC_HMPDIGITWRITERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_HMPDIGITWRITERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/Task.h"
#include "Framework/InputSpec.h"

#include "DetectorsRaw/RawFileReader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "HMPIDBase/Common.h"
#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDReconstruction/HmpidDecoder2.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace hmpid
{

class RawToDigitsTask : public framework::Task
{
 public:
  RawToDigitsTask() = default;
  ~RawToDigitsTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) override;

 private:
  void writeResults();
  void parseNoTF();

  //     static bool eventEquipPadsComparision(o2::hmpid::Digit d1, o2::hmpid::Digit d2);
  std::string mBaseFileName = "";
  std::string mInputRawFileName = "";
  std::string mOutRootFileName = "";

  o2::raw::RawFileReader mReader;
  o2::hmpid::HmpidDecoder2* mDecod;
  std::vector<o2::hmpid::raw::Digit> mAccumulateDigits;
  std::vector<o2::hmpid::raw::Event> mEvents;

  long mDigitsReceived;
  long mFramesReceived;
  long mTotalDigits;
  long mTotalFrames;
  bool mFastAlgorithm;

  ExecutionTimer mExTimer;
};

o2::framework::DataProcessorSpec getRawToDigitsSpec(std::string inputSpec = "HMP/RAWDATA");

} // end namespace hmpid
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_HMPIDDIGITWRITERSPEC_H_ */
