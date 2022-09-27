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

/// @file   TPCResidualReaderSpec.cxx
/// \brief Reads binned residuals which are written by the aggregator and creates the static distortion maps for the TPC
/// \author Ole Schmidt

#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "DataFormatsTPC/Defs.h"
#include "SpacePoints/TrackResiduals.h"
#include "DetectorsBase/Propagator.h"
#include "CommonUtils/StringUtils.h"
#include "TPCInterpolationWorkflow/TPCResidualReaderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class TPCResidualReader : public Task
{
 public:
  TPCResidualReader() = default;
  ~TPCResidualReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTreeResiduals;
  std::unique_ptr<TTree> mTreeStats;
  TrackResiduals mTrackResiduals;
  std::string mFileName = "";
  std::vector<o2::tpc::TrackResiduals::LocalResid> mResiduals, *mResidualsPtr = &mResiduals;
};

void TPCResidualReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("residuals-infile"));
  connectTree(mFileName);
  mTrackResiduals.init();
}

void TPCResidualReader::run(ProcessingContext& pc)
{
  mTrackResiduals.createOutputFile(); // FIXME remove when map output is handled properly
  for (int iSec = 0; iSec < SECTORSPERSIDE * SIDES; ++iSec) {
    auto brStats = mTreeStats->GetBranch(Form("sec%d", iSec));
    brStats->SetAddress(mTrackResiduals.getVoxStatPtr());
    // in case autosave was enabled, we have multiple entries in the statistics tree
    brStats->GetEntry(mTreeStats->GetEntries() - 1); // only the last entry is of interest
    auto brResid = mTreeResiduals->GetBranch(Form("sec%d", iSec));
    brResid->SetAddress(&mResidualsPtr);
    for (int iEntry = 0; iEntry < brResid->GetEntries(); ++iEntry) {
      brResid->GetEntry(iEntry);
      LOGF(debug, "Pushing %lu TPC residuals at entry %i for sector %i", mResiduals.size(), iEntry, iSec);
      for (const auto& res : mResiduals) {
        LOGF(debug, "Adding residual from Voxel %i-%i-%i. dy(%i), dz(%i), tg(%i)", res.bvox[0], res.bvox[1], res.bvox[2], res.dy, res.dz, res.tgSlp);
      }
      mTrackResiduals.getLocalResVec().insert(mTrackResiduals.getLocalResVec().end(), mResiduals.begin(), mResiduals.end());
    }
    // do processing
    mTrackResiduals.processSectorResiduals(iSec);
    // do cleanup
    mTrackResiduals.getLocalResVec().clear();
  }

  mTrackResiduals.closeOutputFile(); // FIXME remove when map output is handled properly

  // const auto& voxResArray = mTrackResiduals.getVoxelResults(); // array with one vector of results per sector
  // pc.outputs().snapshot(Output{"GLO", "VOXELRESULTS", 0, Lifetime::Timeframe}, voxResArray); // send results as one large vector?

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

void TPCResidualReader::connectTree(const std::string& filename)
{
  mTreeResiduals.reset(nullptr); // in case it was already loaded
  mTreeStats.reset(nullptr);     // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTreeResiduals.reset((TTree*)mFile->Get("resid"));
  assert(mTreeResiduals);
  mTreeStats.reset((TTree*)mFile->Get("stats"));
  assert(mTreeStats);

  LOG(info) << "Loaded tree from " << filename << " with " << mTreeResiduals->GetEntries() << " entries";
}

DataProcessorSpec getTPCResidualReaderSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  // outputs.emplace_back("GLO", "VOXELRESULTS", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "tpc-residual-reader",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCResidualReader>()},
    Options{
      {"residuals-infile", VariantType::String, "o2tpc_residuals.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace tpc
} // namespace o2
