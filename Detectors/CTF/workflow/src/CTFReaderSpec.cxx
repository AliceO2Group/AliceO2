// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CTFReaderSpec.cxx

#include <vector>
#include <TFile.h>
#include <TTree.h>

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "CommonUtils/StringUtils.h"
#include "CTFWorkflow/CTFReaderSpec.h"
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsTPC/CTF.h"
#include "Algorithm/RangeTokenizer.h"

using namespace o2::framework;

namespace o2
{
namespace ctf
{

template <typename T>
bool readFromTree(TTree& tree, const std::string brname, T& dest, int ev = 0)
{
  auto* br = tree.GetBranch(brname.c_str());
  if (br && br->GetEntries() > ev) {
    auto* ptr = &dest;
    br->SetAddress(&ptr);
    br->GetEntry(ev);
    br->ResetAddress();
    return true;
  }
  return false;
}

///_______________________________________
CTFReaderSpec::CTFReaderSpec(DetID::mask_t dm, const std::string& inp) : mDets(dm)
{
  mTimer.Stop();
  mTimer.Reset();
  mInput = RangeTokenizer::tokenize<std::string>(inp);
}

///_______________________________________
void CTFReaderSpec::init(InitContext& ic)
{
}

///_______________________________________
void CTFReaderSpec::run(ProcessingContext& pc)
{
  if (mNextToProcess >= mInput.size()) {
    return;
  }

  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  const auto& inputFile = mInput[mNextToProcess];
  LOG(INFO) << "Reading CTF input " << mNextToProcess << ' ' << inputFile;

  TFile flIn(inputFile.c_str());
  if (!flIn.IsOpen() || flIn.IsZombie()) {
    LOG(ERROR) << "Failed to open file " << inputFile;
    throw std::runtime_error("failed to open CTF file");
  }
  std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
  if (!tree) {
    throw std::runtime_error("failed to load CTF tree");
  }
  CTFHeader ctfHeader;
  if (!readFromTree(*tree, "CTFHeader", ctfHeader)) {
    throw std::runtime_error("did not find CTFHeader");
  }
  LOG(INFO) << ctfHeader;

  // send CTF Header
  pc.outputs().snapshot({"header"}, ctfHeader);
  DetID::mask_t detsTF = mDets & ctfHeader.detectors;
  DetID det;

  det = DetID::ITS;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::itsmft::CTF));
    o2::itsmft::CTF::readFromTree(bufVec, *(tree.get()), det.getName());
  }

  det = DetID::MFT;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::itsmft::CTF));
    o2::itsmft::CTF::readFromTree(bufVec, *(tree.get()), det.getName());
  }

  det = DetID::TPC;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::tpc::CTF));
    o2::tpc::CTF::readFromTree(bufVec, *(tree.get()), det.getName());
  }

  mTimer.Stop();
  LOG(INFO) << "Read CTF " << inputFile << " in " << mTimer.CpuTime() - cput << " s";

  if (++mNextToProcess >= mInput.size()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    LOGF(INFO, "CTF reading total timing: Cpu: %.3e Real: %.3e s in %d slots",
         mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  }
}

///_______________________________________
DataProcessorSpec getCTFReaderSpec(DetID::mask_t dets, const std::string& inp)
{
  std::vector<OutputSpec> outputs;

  outputs.emplace_back(OutputLabel{"header"}, "CTF", "HEADER", 0, Lifetime::Timeframe);
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (dets[id]) {
      DetID det(id);
      outputs.emplace_back(OutputLabel{det.getName()}, det.getDataOrigin(), "CTFDATA", 0, Lifetime::Timeframe);
    }
  }
  return DataProcessorSpec{
    "ctf-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<CTFReaderSpec>(dets, inp)},
    Options{}};
}

} // namespace ctf
} // namespace o2
