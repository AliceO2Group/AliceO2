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

#include "CommonConstants/LHCConstants.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include <TFile.h>
#include <THnSparse.h>
#include <fmt/format.h>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

/**
* `o2-mch-rofs-histogrammer` creates a 1D histogram of MCH ROF Records.
*
* The ROFs used are described using the `--rofs-name` option which shouldEnd
* describe the expected DPL input message name (default is `MCH/DIGITROFS`)
*
* The produced 1D histogram is a sparse one (THnSparseL of dimension 1)
* and is written in a Root file named with the `--outfile` option (default
* is `rofs-times.root`).
*
*/

using o2::framework::adaptFromTask;
using o2::framework::AlgorithmSpec;
using o2::framework::ConfigContext;
using o2::framework::ConfigParamSpec;
using o2::framework::ControlService;
using o2::framework::DataProcessorSpec;
using o2::framework::InitContext;
using o2::framework::Inputs;
using o2::framework::Options;
using o2::framework::Outputs;
using o2::framework::ProcessingContext;
using o2::framework::VariantType;
using o2::framework::WorkflowSpec;

struct Histogrammer {
  void init(o2::framework::InitContext& ic)
  {
    mMaxNofTimeFrames = ic.options().get<int>("max-nof-tfs");
    mFirstTF = ic.options().get<int>("first-tf");
    mNofProcessedTFs = 0;
    auto fileName = ic.options().get<std::string>("outfile");
    mHistoFile = std::make_unique<TFile>(fileName.c_str(), "RECREATE");
    if (ic.options().get<bool>("verbose")) {
      fair::Logger::SetConsoleColor(true);
    }
    mFirstOrbit = ic.options().get<int>("first-orbit");
    mLastOrbit = ic.options().get<int>("last-orbit");
  }

  void writeHisto()
  {
    auto firstRof = (mFirstOrbit >= 0) ? o2::InteractionRecord(0, mFirstOrbit) : mRofs.begin()->first.getBCData();

    auto lastRof = (mLastOrbit >= 0) ? o2::InteractionRecord(o2::constants::lhc::LHCMaxBunches, mLastOrbit) : mRofs.rbegin()->first.getBCData();
    auto nbins = static_cast<Int_t>(lastRof.differenceInBC(firstRof));
    Int_t bins[1] = {nbins};
    Double_t xmin[1] = {0.0};
    Double_t xmax[1] = {nbins * 1.0 + 1};
    THnSparseL h("rof_times", "rof times", 1, bins, xmin, xmax);
    for (const auto& p : mRofs) {
      const auto& rof = p.first;
      Double_t w = rof.getNEntries();
      Double_t x[1] = {1.0 * rof.getBCData().differenceInBC(firstRof)};
      for (auto i = 0; i < rof.getBCWidth(); i++) {
        x[0] += 1.0;
        h.Fill(x, w);
        h.SetBinError(h.GetBin(x), sqrt(w));
      }
    }
    h.Write("", TObject::kWriteDelete);
  }

  void run(ProcessingContext& pc)
  {
    bool shouldEnd = mNofProcessedTFs >= mMaxNofTimeFrames;
    if (shouldEnd) {
      pc.services().get<ControlService>().endOfStream();
      return;
    }

    bool shouldProcess = (mTFid >= mFirstTF && mNofProcessedTFs < mMaxNofTimeFrames);
    if (!shouldProcess) {
      return;
    }

    mTFid++;
    auto rofs = pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("rofs");
    for (const auto& rof : rofs) {
      mRofs[rof] += rof.getNEntries();
    }
    writeHisto();
  }
  size_t mFirstTF{0};                                           // first timeframe to process
  size_t mMaxNofTimeFrames{std::numeric_limits<size_t>::max()}; // max number of timeframes to process
  size_t mNofProcessedTFs{0};                                   // actual number of timeframes processed so far
  size_t mTFid{0};                                              // current timeframe index
  std::unique_ptr<TFile> mHistoFile;                            // output histogram file
  std::map<o2::mch::ROFRecord, int> mRofs;                      // accumulation of rofs
  uint32_t mFirstOrbit;
  uint16_t mLastOrbit;
};

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"rofs-name", VariantType::String, "MCH/DIGITROFS", {"name of the input rofs"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  WorkflowSpec specs;

  auto rofName = cc.options().get<std::string>("rofs-name");
  std::string inputConfig = "rofs:" + rofName;

  DataProcessorSpec rofHistogrammer{
    "mch-rofs-histogrammer",
    Inputs{o2::framework::select(inputConfig.c_str())},
    Outputs{},
    AlgorithmSpec{adaptFromTask<Histogrammer>()},
    Options{
      {"verbose", VariantType::Bool, false, {"verbose output"}},
      {"max-nof-tfs", VariantType::Int, 10, {"max number of timeframes to process"}},
      {"first-tf", VariantType::Int, 0, {"first timeframe to process"}},
      {"first-orbit", VariantType::Int, 0, {"force first orbit to use as first orbit (for histogram) (default=-1=auto)"}},
      {"last-orbit", VariantType::Int, 0, {"force last orbit to use as last orbit (for histogram) (default=-1=auto)"}},
      {"outfile", VariantType::String, "rofs-times.root", {"name of the histogram output file"}}}};

  specs.push_back(rofHistogrammer);
  return specs;
}
