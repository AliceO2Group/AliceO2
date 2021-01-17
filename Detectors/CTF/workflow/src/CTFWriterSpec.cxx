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
#include <TSystem.h>

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "CTFWorkflow/CTFWriterSpec.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsFT0/CTF.h"
#include "DataFormatsFV0/CTF.h"
#include "DataFormatsFDD/CTF.h"
#include "DataFormatsTOF/CTF.h"
#include "DataFormatsMID/CTF.h"
#include "DataFormatsEMCAL/CTF.h"
#include "DataFormatsPHOS/CTF.h"
#include "DataFormatsCPV/CTF.h"

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

CTFWriterSpec::CTFWriterSpec(DetID::mask_t dm, uint64_t r, bool doCTF, bool doDict, bool dictPerDet)
  : mDets(dm), mRun(r), mWriteCTF(doCTF), mCreateDict(doDict), mDictPerDetector(dictPerDet)
{
  mTimer.Stop();
  mTimer.Reset();

  if (doDict) { // make sure that there is no local dictonary
    for (int id = 0; id < DetID::nDetectors; id++) {
      DetID det(id);
      if (isPresent(det)) {
        auto dictName = dictionaryFileName(det.getName());
        if (gSystem->AccessPathName(dictName.c_str()) == 0) {
          throw std::runtime_error(o2::utils::concat_string("CTF dictionary creation is requested but ", dictName, " already exists, remove it!"));
        }
        if (!mDictPerDetector) {
          break; // no point in checking further
        }
      }
    }
  }
}

void CTFWriterSpec::init(InitContext& ic)
{
  mSaveDictAfter = ic.options().get<int>("save-dict-after");
}

void CTFWriterSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto tfOrb = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0))->firstTForbit;

  std::unique_ptr<TFile> fileOut;
  std::unique_ptr<TTree> treeOut;
  if (mWriteCTF) {
    //    fileOut.reset(TFile::Open(o2::base::NameConf::getCTFFileName(tfOrb).c_str(), "recreate"));
    // RS Until the DPL will propagate the firstTForbit, we will use simple counter in CTF file name to avoid overwriting in case of multiple TFs
    fileOut.reset(TFile::Open(o2::base::NameConf::getCTFFileName(mNTF).c_str(), "recreate"));
    treeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
  }

  // create header
  CTFHeader header{mRun, tfOrb};

  processDet<o2::itsmft::CTF>(pc, DetID::ITS, header, treeOut.get());
  processDet<o2::itsmft::CTF>(pc, DetID::MFT, header, treeOut.get());
  processDet<o2::tpc::CTF>(pc, DetID::TPC, header, treeOut.get());
  processDet<o2::tof::CTF>(pc, DetID::TOF, header, treeOut.get());
  processDet<o2::ft0::CTF>(pc, DetID::FT0, header, treeOut.get());
  processDet<o2::fv0::CTF>(pc, DetID::FV0, header, treeOut.get());
  processDet<o2::fdd::CTF>(pc, DetID::FDD, header, treeOut.get());
  processDet<o2::mid::CTF>(pc, DetID::MID, header, treeOut.get());
  processDet<o2::emcal::CTF>(pc, DetID::EMC, header, treeOut.get());
  processDet<o2::phos::CTF>(pc, DetID::PHS, header, treeOut.get());
  processDet<o2::cpv::CTF>(pc, DetID::CPV, header, treeOut.get());

  mTimer.Stop();

  if (mWriteCTF) {
    appendToTree(*treeOut.get(), "CTFHeader", header);
    treeOut->SetEntries(1);
    treeOut->Write();
    treeOut.reset();
    fileOut->Close();
    LOG(INFO) << "TF#" << mNTF << ": wrote " << fileOut->GetName() << " with CTF{" << header << "} in " << mTimer.CpuTime() - cput << " s";
  } else {
    LOG(INFO) << "TF#" << mNTF << " CTF writing is disabled";
  }
  mNTF++;
  if (mCreateDict && mSaveDictAfter > 0 && (mNTF % mSaveDictAfter) == 0) {
    storeDictionaries();
  }
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
      mDictTreeOut->SetEntries(1);
      mDictTreeOut->Write();
      mDictTreeOut.reset();
      mDictFileOut.reset();
    }
  }
  if (!mDictTreeOut) {
    mDictFileOut.reset(TFile::Open(dictionaryFileName(det.getName()).c_str(), "recreate"));
    mDictTreeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFDICT).c_str(), "O2 CTF dictionary");
  }
}

std::string CTFWriterSpec::dictionaryFileName(const std::string& detName)
{
  if (mDictPerDetector) {
    if (detName.empty()) {
      throw std::runtime_error("Per-detector dictionary files are requested but detector name is not provided");
    }
    return o2::utils::concat_string(detName, "_", o2::base::NameConf::CTFDICT, ".root");
  } else {
    return o2::utils::concat_string(o2::base::NameConf::CTFDICT, ".root");
  }
}

void CTFWriterSpec::storeDictionaries()
{
  CTFHeader header{mRun, uint32_t(mNTF)};
  storeDictionary<o2::itsmft::CTF>(DetID::ITS, header);
  storeDictionary<o2::itsmft::CTF>(DetID::MFT, header);
  storeDictionary<o2::tpc::CTF>(DetID::TPC, header);
  storeDictionary<o2::tof::CTF>(DetID::TOF, header);
  storeDictionary<o2::ft0::CTF>(DetID::FT0, header);
  storeDictionary<o2::fv0::CTF>(DetID::FV0, header);
  storeDictionary<o2::fdd::CTF>(DetID::FDD, header);
  storeDictionary<o2::mid::CTF>(DetID::MID, header);
  storeDictionary<o2::emcal::CTF>(DetID::EMC, header);
  storeDictionary<o2::phos::CTF>(DetID::PHS, header);
  storeDictionary<o2::cpv::CTF>(DetID::CPV, header);
  // close remnants
  if (mDictTreeOut) {
    closeDictionaryTreeAndFile(header);
  }
  LOG(INFO) << "Saved CTF dictionary after " << mNTF << " TFs processed";
}

void CTFWriterSpec::closeDictionaryTreeAndFile(CTFHeader& header)
{
  if (mDictTreeOut) {
    appendToTree(*mDictTreeOut.get(), "CTFHeader", header);
    mDictTreeOut->SetEntries(1);
    mDictTreeOut->Write(mDictTreeOut->GetName(), TObject::kSingleKey);
    mDictTreeOut.reset();
    mDictFileOut.reset();
  }
}

DataProcessorSpec getCTFWriterSpec(DetID::mask_t dets, uint64_t run, bool doCTF, bool doDict, bool dictPerDet)
{
  std::vector<InputSpec> inputs;
  LOG(INFO) << "Detectors list:";
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
    Options{{"save-dict-after", VariantType::Int, -1, {"In dictionary generation mode save it dictionary after certain number of TFs processed"}}}};
}

} // namespace ctf
} // namespace o2
