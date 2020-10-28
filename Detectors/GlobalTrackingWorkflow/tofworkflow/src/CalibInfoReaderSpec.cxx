// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.cxx

#include <vector>
#include <unistd.h>

#include "TChain.h"
#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "TOFWorkflow/CalibInfoReaderSpec.h"

using namespace o2::framework;
using namespace o2::tof;

namespace o2
{
namespace tof
{

void CalibInfoReader::init(InitContext& ic)
{
  LOG(INFO) << "Init CalibInfo reader!";
  mFile = fopen(mFileName, "r");
  if (!mFile) {
    LOG(ERROR) << "Cannot open the " << mFileName << " file !";
    mState = 0;
    return;
  }
  mState = 1;
}

void CalibInfoReader::run(ProcessingContext& pc)
{
  if (mState != 1) {
    return;
  }

  char filename[100];

  if ((mTree && mCurrentEntry < mTree->GetEntries()) || fscanf(mFile, "%s", filename) == 1) {
    if (!mTree || mCurrentEntry >= mTree->GetEntries()) {
      TFile* fin = TFile::Open(filename);
      mTree = (TTree*)fin->Get("calibTOF");
      mCurrentEntry = 0;
      mTree->SetBranchAddress("TOFCalibInfo", &mPvect);
    }
    if ((mGlobalEntry % mNinstances) == mInstance) {
      mTree->GetEvent(mCurrentEntry);
      LOG(INFO) << "Send " << mVect.size() << " calib infos";
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe}, mVect);
      usleep(10000);
    }
    mGlobalEntry++;
    mCurrentEntry++;
  } else {
    mState = 2;
    pc.services().get<ControlService>().endOfStream();
  }
  return;
}

DataProcessorSpec getCalibInfoReaderSpec(int instance, int ninstances, const char* filename)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe);

  const char* nameSpec;
  if (ninstances == 1) {
    nameSpec = "tof-calibinfo-reader";
  } else {
    nameSpec = Form("tof-calibinfo-reader-%d", instance);
  }

  return DataProcessorSpec{
    nameSpec,
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<CalibInfoReader>(instance, ninstances, filename)},
    Options{/* for the moment no options */}};
}
} // namespace tof
} // namespace o2
