// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TRDCalibReaderSpec.cxx

#include "TRDWorkflowIO/TRDCalibReaderSpec.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "fairlogger/Logger.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

void TRDCalibReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(INFO) << "Init TRD tracklet reader!";
  mInFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("trd-calib-infile"));
  mInTreeName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("treename"));
  connectTree();
}

void TRDCalibReader::connectTree()
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(mInFileName.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mInTreeName.c_str()));
  assert(mTree);
  mTree->SetBranchAddress("AngularResids", &mAngResidPtr);
  LOG(INFO) << "Loaded tree from " << mInFileName << " with " << mTree->GetEntries() << " entries";
}

void TRDCalibReader::run(ProcessingContext& pc)
{
  auto currEntry = mTree->GetReadEntry() + 1;
  assert(currEntry < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(currEntry);
  if (mAngResids.size() > 0) {
    LOG(INFO) << "Pushing angular residual histograms filled with " << mAngResids.at(0).getNEntries() << " entries at tree entry " << currEntry;
  } else {
    LOG(WARNING) << "No TRD calibration data available in the tree";
  }
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "ANGRESHISTS", 0, Lifetime::Timeframe}, mAngResids);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getTRDCalibReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTRD, "ANGRESHISTS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TRDCalibReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDCalibReader>()},
    Options{
      {"trd-calib-infile", VariantType::String, "trdangreshistos.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "calibdata", {"Name of top-level TTree"}},
    }};
}

} // namespace trd
} // namespace o2
