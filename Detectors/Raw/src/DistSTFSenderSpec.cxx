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

#include "DetectorsRaw/DistSTFSenderSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include "Headers/STFHeader.h"

using namespace o2::framework;
using namespace o2::header;

namespace o2::raw
{

class DistSTFSender : public Task
{
 public:
  DistSTFSender(int m, unsigned subsp) : mMaxTF(m), mSubSpec(subsp) {}
  void run(ProcessingContext& ctx) final;

 private:
  int mMaxTF = 1;
  unsigned mSubSpec = 0;
  int mTFCount = 0;
};

//___________________________________________________________
void DistSTFSender::run(ProcessingContext& pc)
{
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  STFHeader stfHeader{tinfo.tfCounter, tinfo.firstTForbit, tinfo.runNumber};
  pc.outputs().snapshot(o2::framework::Output{gDataOriginFLP, gDataDescriptionDISTSTF, mSubSpec}, stfHeader);
  if (++mTFCount >= mMaxTF) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

//_________________________________________________________
DataProcessorSpec getDistSTFSenderSpec(int maxTF, unsigned subsp)
{
  return DataProcessorSpec{
    "dist-stf-sender",
    Inputs{},
    Outputs{OutputSpec{gDataOriginFLP, gDataDescriptionDISTSTF, subsp}},
    AlgorithmSpec{adaptFromTask<DistSTFSender>(maxTF, subsp)},
    Options{}};
}

} // namespace o2::raw
