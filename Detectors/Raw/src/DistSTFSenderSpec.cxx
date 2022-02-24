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
  DistSTFSender(int m) : mMaxTF(m) {}
  void run(ProcessingContext& ctx) final;

 private:
  int mMaxTF = 1;
  int mTFCount = 0;
};

//___________________________________________________________
void DistSTFSender::run(ProcessingContext& pc)
{
  const auto* dh = DataRefUtils::getHeader<DataHeader*>(pc.inputs().getFirstValid(true));
  auto creationTime = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;
  LOG(info) << "DIST STF for run " << dh->runNumber << " orbit " << dh->firstTForbit << " creation " << creationTime << " TFid: " << dh->tfCounter;
  STFHeader stfHeader{dh->tfCounter, dh->firstTForbit, dh->runNumber};
  pc.outputs().snapshot(o2::framework::Output{gDataOriginFLP, gDataDescriptionDISTSTF, 0}, stfHeader);
  if (++mTFCount >= mMaxTF) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

//_________________________________________________________
DataProcessorSpec getDistSTFSenderSpec(int maxTF)
{
  return DataProcessorSpec{
    "dist-stf-sender",
    Inputs{},
    Outputs{OutputSpec{gDataOriginFLP, gDataDescriptionDISTSTF, 0}},
    AlgorithmSpec{adaptFromTask<DistSTFSender>(maxTF)},
    Options{}};
}

} // namespace o2::raw
