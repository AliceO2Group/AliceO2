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

/// @file   DigitReaderSpec.cxx

#include <vector>
#include <unistd.h>

#include "TChain.h"
#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "TOFWorkflowIO/CalibInfoReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::tof;

namespace o2
{
namespace tof
{

constexpr o2::header::DataDescription ddCalib{"CALIBDATA"}, ddCalib_tpc{"CALIBDATA_TPC"}, ddDia{"DIAFREQ"};

void CalibInfoReader::init(InitContext& ic)
{
  mDiagnostic = ic.options().get<bool>("enable-dia");
  LOG(debug) << "Init CalibInfo reader!";
  auto fname = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), mFileName);
  mFile = fopen(fname.c_str(), "r");
  if (!mFile) {
    LOG(error) << "Cannot open the " << fname << " file !";
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
  auto& timingInfo = pc.services().get<o2::framework::TimingInfo>();
  char filename[100];

  if ((mTree && mCurrentEntry < mTree->GetEntries()) || fscanf(mFile, "%s", filename) == 1) {
    if (!mTree || mCurrentEntry >= mTree->GetEntries()) {
      TFile* fin = TFile::Open(filename);
      mTree = (TTree*)fin->Get("calibTOF");
      mCurrentEntry = 0;
      mTree->SetBranchAddress("TOFCalibInfo", &mPvect);

      if (mDiagnostic) {
        mTree->SetBranchAddress("TOFDiaInfo", &mPdia);
      }

      LOG(debug) << "Open " << filename;
    }
    if ((mGlobalEntry % mNinstances) == mInstance) {
      mTree->GetEvent(mCurrentEntry);

      if (mDiagnostic) {
        // add TFIDInfo
        const auto& info = mDia.getTFIDInfo();
        timingInfo.firstTForbit = info.firstTForbit;
        timingInfo.tfCounter = info.tfCounter;
        timingInfo.runNumber = info.runNumber;
        timingInfo.timeslice = info.startTime;
        timingInfo.creation = info.creation;
      } else if (mVect.size()) {
        timingInfo.creation = uint64_t(mVect[0].getTimestamp()) * 1000;
      }

      LOG(debug) << "Current entry " << mCurrentEntry;
      LOG(debug) << "Send " << mVect.size() << " calib infos";
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, mTOFTPC ? ddCalib_tpc : ddCalib, 0, Lifetime::Timeframe}, mVect);

      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, ddDia, 0, Lifetime::Timeframe}, mDia);

      usleep(100);
    }
    mGlobalEntry++;
    mCurrentEntry++;
  } else {
    mState = 2;
    pc.services().get<ControlService>().endOfStream();
  }
  return;
}

DataProcessorSpec getCalibInfoReaderSpec(int instance, int ninstances, const char* filename, bool toftpc)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, toftpc ? ddCalib_tpc : ddCalib, 0, Lifetime::Timeframe);

  std::string nameSpec = "tof-calibinfo-reader";
  if (toftpc) {
    nameSpec += "-tpc";
  }
  if (ninstances > 1) {
    nameSpec += fmt::format("-{:d}", instance);
  }

  outputs.emplace_back(o2::header::gDataOriginTOF, ddDia, 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    nameSpec,
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<CalibInfoReader>(instance, ninstances, filename)},
    Options{{"input-dir", VariantType::String, "none", {"Input directory"}}, {"enable-dia", VariantType::Bool, false, {"read also diagnostic frequency"}}}};
}
} // namespace tof
} // namespace o2
