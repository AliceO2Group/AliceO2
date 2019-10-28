// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFCalibWriterSpec.cxx

#include "TOFWorkflow/TOFCalibWriterSpec.h"
#include "Framework/ControlService.h"
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/Cluster.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{
using evIdx = o2::dataformats::EvIndex<int, int>;
using OutputType = std::vector<o2::dataformats::CalibInfoTOF>;

template <typename T>
TBranch* getOrMakeBranch(TTree& tree, std::string brname, T* ptr)
{
  if (auto br = tree.GetBranch(brname.c_str())) {
    br->SetAddress(static_cast<void*>(&ptr));
    return br;
  }
  // otherwise make it
  return tree.Branch(brname.c_str(), ptr);
}

void TOFCalibWriter::init(InitContext& ic)
{
  // get the option from the init context
  mOutFileName = ic.options().get<std::string>("tof-calib-outfile");
  mOutTreeName = ic.options().get<std::string>("treename");
}

void TOFCalibWriter::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }

  TFile outf(mOutFileName.c_str(), "recreate");
  if (outf.IsZombie()) {
    LOG(FATAL) << "Failed to open output file " << mOutFileName;
  }
  TTree tree(mOutTreeName.c_str(), "Tree of TOF calib infos");
  auto indata = pc.inputs().get<OutputType>("tofcalibinfo");
  LOG(INFO) << "RECEIVED MATCHED SIZE " << indata.size();

  auto br = getOrMakeBranch(tree, "TOFCalibInfo", &indata);
  br->Fill();

  tree.SetEntries(1);
  tree.Write();
  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getTOFCalibWriterSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofcalibinfo", "TOF", "CALIBINFOS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFCalibWriter",
    inputs,
    {}, // no output
    AlgorithmSpec{adaptFromTask<TOFCalibWriter>()},
    Options{
      {"tof-calib-outfile", VariantType::String, "o2calib_tof.root", {"Name of the input file"}},
      {"treename", VariantType::String, "calibTOF", {"Name of top-level TTree"}},
    }};
}
} // namespace tof
} // namespace o2
