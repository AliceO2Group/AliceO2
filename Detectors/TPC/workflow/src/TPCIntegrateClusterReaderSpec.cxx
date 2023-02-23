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

/// @file   TPCIntegrateClusterReaderSpec.cxx

#include <vector>
#include <boost/algorithm/string/predicate.hpp>

#include "Framework/DeviceSpec.h"
#include "TPCWorkflow/TPCIntegrateClusterReaderSpec.h"
#include "DetectorsCalibration/IntegratedClusterCalibrator.h"
#include "TPCWorkflow/TPCIntegrateClusterSpec.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonUtils/NameConf.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "Algorithm/RangeTokenizer.h"
#include "TChain.h"
#include "TGrid.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class IntegratedClusterReader : public Task
{
 public:
  IntegratedClusterReader() = default;
  ~IntegratedClusterReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTrees();
  bool checkEOS(ProcessingContext& pc);

  int mLane = 0;
  int mNLanes = 1;
  int mChainEntry = 0;                                       ///< processed entries in the chain
  std::unique_ptr<TChain> mChain;                            ///< input TChain
  std::vector<std::string> mFileNames;                       ///< input files
  ITPCC mTPCC, *mTPCCPtr = &mTPCC;                           ///< branch integrated number of cluster TPC currents
  o2::dataformats::TFIDInfo mTFinfo, *mTFinfoPtr = &mTFinfo; ///< branch TFIDInfo for injecting correct time
  std::vector<std::pair<unsigned long, int>> mIndices;       ///< firstTfOrbit, file, index
};

void IntegratedClusterReader::init(InitContext& ic)
{
  mLane = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  mChainEntry = mLane;
  mNLanes = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;

  const auto dontCheckFileAccess = ic.options().get<bool>("dont-check-file-access");
  auto fileList = o2::RangeTokenizer::tokenize<std::string>(ic.options().get<std::string>("tpc-currents-infiles"));

  // check if only one input file (a txt file contaning a list of files is provided)
  if (fileList.size() == 1) {
    if (boost::algorithm::ends_with(fileList.front(), "txt")) {
      LOGP(info, "Reading files from input file {}", fileList.front());
      std::ifstream is(fileList.front());
      std::istream_iterator<std::string> start(is);
      std::istream_iterator<std::string> end;
      std::vector<std::string> fileNamesTmp(start, end);
      fileList = fileNamesTmp;
    }
  }

  const std::string inpDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir"));
  for (const auto& file : fileList) {
    if ((file.find("alien://") == 0) && !gGrid && !TGrid::Connect("alien://")) {
      LOG(fatal) << "Failed to open alien connection";
    }
    const auto fileDir = o2::utils::Str::concat_string(inpDir, file);
    if (!dontCheckFileAccess) {
      std::unique_ptr<TFile> filePtr(TFile::Open(fileDir.data()));
      if (!filePtr || !filePtr->IsOpen() || filePtr->IsZombie()) {
        LOGP(warning, "Could not open file {}", fileDir);
        continue;
      }
    }
    mFileNames.emplace_back(fileDir);
  }

  if (mFileNames.size() == 0) {
    LOGP(error, "No input files to process");
  }
  connectTrees();
}

void IntegratedClusterReader::run(ProcessingContext& pc)
{
  // check time order inside the TChain
  if (mChainEntry == mLane) {
    mIndices.clear();
    mIndices.reserve(mChain->GetEntries());
    // disable all branches except the firstTForbit branch to significantly speed up the loop over the TTree
    mChain->SetBranchStatus("*", 0);
    mChain->SetBranchStatus("firstTForbit", 1);
    for (unsigned long i = 0; i < mChain->GetEntries(); i++) {
      mChain->GetEntry(i);
      mIndices.emplace_back(std::make_pair(mTFinfo.firstTForbit, i));
    }
    mChain->SetBranchStatus("*", 1);
    std::sort(mIndices.begin(), mIndices.end());
    if (checkEOS(pc)) {
      return;
    }
  }

  LOGP(debug, "Processing entry {}", mIndices[mChainEntry].second);
  mChain->GetEntry(mIndices[mChainEntry].second);
  mChainEntry += mNLanes;

  // inject correct timing informations
  auto& timingInfo = pc.services().get<o2::framework::TimingInfo>();
  timingInfo.firstTForbit = mTFinfo.firstTForbit;
  timingInfo.tfCounter = mTFinfo.tfCounter;
  timingInfo.runNumber = mTFinfo.runNumber;
  timingInfo.creation = mTFinfo.creation;

  pc.outputs().snapshot(Output{header::gDataOriginTPC, getDataDescriptionTPCC()}, mTPCC);
  usleep(100);

  checkEOS(pc);
}

void IntegratedClusterReader::connectTrees()
{
  mChain.reset(new TChain("itpcc"));
  for (const auto& file : mFileNames) {
    LOGP(info, "Adding file to chain: {}", file);
    mChain->AddFile(file.data());
  }
  assert(mChain->GetEntries());
  mChain->SetBranchAddress("ITPCC", &mTPCCPtr);
  mChain->SetBranchAddress("tfID", &mTFinfoPtr);
}

bool IntegratedClusterReader::checkEOS(ProcessingContext& pc)
{
  if (mChainEntry >= mChain->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return true;
  }
  return false;
}

DataProcessorSpec getTPCIntegrateClusterReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTPC, getDataDescriptionTPCC(), 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-integrated-cluster-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<IntegratedClusterReader>()},
    Options{
      {"tpc-currents-infiles", VariantType::String, "o2currents_tpc.root", {"comma-separated list of input files or .txt file containing list of input files"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"dont-check-file-access", VariantType::Bool, false, {"Deactivate check if all files are accessible before adding them to the list of files"}},
    }};
}

} // namespace tpc
} // namespace o2
