// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CTFWriterSpec.cxx

#include <vector>
#include <TFile.h>
#include <TTree.h>

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "CTFWorkflow/CTFWriterSpec.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsFT0/CTF.h"
#include "DataFormatsTOF/CTF.h"

using namespace o2::framework;

namespace o2
{
namespace ctf
{

template <typename T>
void appendToTree(TTree& tree, const std::string brname, T& ptr)
{
  auto* br = tree.GetBranch(brname.c_str());
  auto* pptr = &ptr;
  if (br) {
    br->SetAddress(&pptr);
  } else {
    br = tree.Branch(brname.c_str(), &pptr);
  }
  br->Fill();
  br->ResetAddress();
}

void CTFWriterSpec::init(InitContext& ic)
{
}

void CTFWriterSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto tfOrb = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0))->firstTForbit;

  std::unique_ptr<TFile> fileOut;
  std::unique_ptr<TTree> treeOut;
  if (mWriteCTF) {
    fileOut.reset(TFile::Open(o2::base::NameConf::getCTFFileName(tfOrb).c_str(), "recreate"));
    treeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
  }

  // create header
  CTFHeader header{mRun, tfOrb};

  processDet<o2::itsmft::CTF>(pc, DetID::ITS, header, treeOut.get());
  processDet<o2::itsmft::CTF>(pc, DetID::MFT, header, treeOut.get());
  processDet<o2::tpc::CTF>(pc, DetID::TPC, header, treeOut.get());
  processDet<o2::tof::CTF>(pc, DetID::TOF, header, treeOut.get());
  processDet<o2::ft0::CTF>(pc, DetID::FT0, header, treeOut.get());

  mTimer.Stop();

  if (mWriteCTF) {
    appendToTree(*treeOut.get(), "CTFHeader", header);
    treeOut->SetEntries(1);
    treeOut->Write();
    treeOut.reset();
    fileOut->Close();
    LOG(INFO) << "TF#" << mNTF << ": wrote " << fileOut->GetName() << " with CTF{" << header << "} in " << mTimer.CpuTime() - cput << " s";
  }
  mNTF++;
}

void CTFWriterSpec::endOfStream(EndOfStreamContext& ec)
{

  if (mCreateDict) {
    storeDictionaries();
  }

  LOGF(INFO, "CTF writing total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void CTFWriterSpec::prepareDictionaryTreeAndFile(DetID det)
{
  if (mDictPerDetector) {
    if (mDictTreeOut) {
      mDictTreeOut->Write();
      mDictTreeOut.reset();
      mDictFileOut.reset();
    }
  }
  if (!mDictTreeOut) {
    std::string fnm = mDictPerDetector ? o2::utils::concat_string(det.getName(), "_", o2::base::NameConf::CTFDICT, ".root") : o2::utils::concat_string(o2::base::NameConf::CTFDICT, ".root");
    mDictFileOut.reset(TFile::Open(fnm.c_str(), "recreate"));
    mDictTreeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFDICT).c_str(), "O2 CTF dictionary");
  }
}

void CTFWriterSpec::storeDictionaries()
{
  CTFHeader header{mRun, 0};
  storeDictionary<o2::itsmft::CTF>(DetID::ITS, header);
  storeDictionary<o2::itsmft::CTF>(DetID::MFT, header);
  storeDictionary<o2::tpc::CTF>(DetID::TPC, header);
  storeDictionary<o2::tof::CTF>(DetID::TOF, header);
  storeDictionary<o2::ft0::CTF>(DetID::FT0, header);
  // close remnants
  if (mDictTreeOut) {
    mDictTreeOut->SetEntries(1);
    appendToTree(*mDictTreeOut.get(), "CTFHeader", header);
    mDictTreeOut->Write();
    mDictTreeOut.reset();
    mDictFileOut.reset();
  }
}

DataProcessorSpec getCTFWriterSpec(DetID::mask_t dets, uint64_t run, bool doCTF, bool doDict, bool dictPerDet)
{
  std::vector<InputSpec> inputs;
  LOG(INFO) << "Det list:";
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (dets[id]) {
      inputs.emplace_back(DetID::getName(id), DetID::getDataOrigin(id), "CTFDATA", 0, Lifetime::Timeframe);
      LOG(INFO) << "Det " << DetID::getName(id) << " added";
    }
  }
  return DataProcessorSpec{
    "ctf-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<CTFWriterSpec>(dets, run, doCTF, doDict, dictPerDet)},
    Options{}};
}

} // namespace ctf
} // namespace o2
