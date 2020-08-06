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

#include "TChain.h"
#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
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
  int ientry = 0;
  while (fscanf(mFile, "%s", filename) == 1) {
    TFile* fin = TFile::Open(filename);
    TTree* tin = (TTree*)fin->Get("calibTOF");
    tin->SetBranchAddress("TOFCalibInfo", &mPvect);
    for (int i = 0; i < tin->GetEntries(); i += mNinstances) {
      if ((ientry % mNinstances) == mInstance) {
        tin->GetEvent(i);
        LOG(INFO) << "Send " << mVect.size() << " calib infos";
        pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CALIBINFOS", 0, Lifetime::Timeframe}, mVect);
      }
      ientry++;
    }
  }

  mState = 2;
  pc.services().get<ControlService>().endOfStream();
  return;
}

DataProcessorSpec getCalibInfoReaderSpec(int instance, int ninstances, const char* filename)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "CALIBINFOS", 0, Lifetime::Timeframe);

  const char* nameSpec;
  if (ninstances == 1)
    nameSpec = "tof-calibinfo-reader";
  else
    nameSpec = Form("tof-calibinfo-reader-%d", instance);

  return DataProcessorSpec{
    nameSpec,
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<CalibInfoReader>(instance, ninstances, filename)},
    Options{/* for the moment no options */}};
}
} // namespace tof
} // namespace o2
