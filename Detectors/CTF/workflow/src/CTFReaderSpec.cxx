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
#include "DataFormatsTRD/CTF.h"
#include "DataFormatsFT0/CTF.h"
#include "DataFormatsFV0/CTF.h"
#include "DataFormatsFDD/CTF.h"
#include "DataFormatsTOF/CTF.h"
#include "DataFormatsMID/CTF.h"
#include "DataFormatsMCH/CTF.h"
#include "DataFormatsEMCAL/CTF.h"
#include "DataFormatsPHOS/CTF.h"
#include "DataFormatsCPV/CTF.h"
#include "DataFormatsZDC/CTF.h"
#include "DataFormatsHMP/CTF.h"
#include "Algorithm/RangeTokenizer.h"
#include <TStopwatch.h>

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

using DetID = o2::detectors::DetID;

class CTFReaderSpec : public o2::framework::Task
{
 public:
  CTFReaderSpec(DetID::mask_t dm, const std::string& inp, int loop = 1, int delayMUS = 0);
  ~CTFReaderSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void openCTFFile(const std::string& flname);

  DetID::mask_t mDets;             // detectors
  std::vector<std::string> mInput; // input files
  std::unique_ptr<TFile> mCTFFile;
  std::unique_ptr<TTree> mCTFTree;
  uint32_t mCTFCounter = 0;
  size_t mNextToProcess = 0;
  int mCurrEntry = 0;
  int mLoops = 1;
  int mLoopsCounter = 0;
  int mDelayMUS = 0;
  std::string mCTFDir = "";
  TStopwatch mTimer;
};

///_______________________________________
CTFReaderSpec::CTFReaderSpec(DetID::mask_t dm, const std::string& inp, int loop, int delayMUS) : mDets(dm), mLoops(loop), mDelayMUS(delayMUS)
{
  mTimer.Stop();
  mTimer.Reset();
  mInput = RangeTokenizer::tokenize<std::string>(inp);
}

///_______________________________________
void CTFReaderSpec::init(InitContext& ic)
{
  mCTFDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir"));
}

///_______________________________________
void CTFReaderSpec::openCTFFile(const std::string& flname)
{
  mCTFFile.reset(TFile::Open(flname.c_str()));
  if (!mCTFFile->IsOpen() || mCTFFile->IsZombie()) {
    LOG(ERROR) << "Failed to open file " << flname;
    throw std::runtime_error("failed to open CTF file");
  }
  mCTFTree.reset((TTree*)mCTFFile->Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
  if (!mCTFTree) {
    throw std::runtime_error("failed to load CTF tree");
  }
  mCurrEntry = 0;
}

///_______________________________________
void CTFReaderSpec::run(ProcessingContext& pc)
{
  if (mNextToProcess >= mInput.size()) {
    return;
  }
  if (mDelayMUS && mCTFCounter > 0) {
    usleep(mDelayMUS);
  }

  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  if (!mCTFTree) { // there is still a tree open with multiple entries
    std::string inputFile = o2::utils::Str::concat_string(mCTFDir, mInput[mNextToProcess]);
    LOG(INFO) << "Reading CTF input " << mNextToProcess << ' ' << inputFile;
    openCTFFile(inputFile);
  }
  CTFHeader ctfHeader;
  if (!readFromTree(*(mCTFTree.get()), "CTFHeader", ctfHeader, mCurrEntry)) {
    throw std::runtime_error("did not find CTFHeader");
  }
  LOG(INFO) << ctfHeader;

  auto setFirstTFOrbit = [&pc, &ctfHeader, this](const std::string& label) {
    auto* hd = pc.outputs().findMessageHeader({label});
    if (!hd) {
      throw std::runtime_error(o2::utils::Str::concat_string("failed to find output message header for ", label));
    }
    hd->firstTForbit = ctfHeader.firstTForbit;
    hd->tfCounter = this->mCTFCounter;
  };

  // send CTF Header
  pc.outputs().snapshot({"header"}, ctfHeader);
  setFirstTFOrbit("header");

  DetID::mask_t detsTF = mDets & ctfHeader.detectors;
  DetID det;

  det = DetID::ITS;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::itsmft::CTF));
    o2::itsmft::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::MFT;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::itsmft::CTF));
    o2::itsmft::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::TPC;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::tpc::CTF));
    o2::tpc::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::TRD;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::trd::CTF));
    o2::trd::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::FT0;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::ft0::CTF));
    o2::ft0::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::FV0;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::fv0::CTF));
    o2::fv0::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::FDD;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::fdd::CTF));
    o2::fdd::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::TOF;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::tof::CTF));
    o2::tof::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::MID;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::mid::CTF));
    o2::mid::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::MCH;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::mch::CTF));
    o2::mch::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::EMC;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::emcal::CTF));
    o2::emcal::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::PHS;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::phos::CTF));
    o2::phos::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::CPV;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::cpv::CTF));
    o2::cpv::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::ZDC;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::zdc::CTF));
    o2::zdc::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::HMP;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::hmpid::CTF));
    o2::hmpid::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrEntry);
    setFirstTFOrbit(det.getName());
  }

  mTimer.Stop();
  LOGP(INFO, "Read CTF#{} ({} of {} in {}) in {:.3f} s", mCTFCounter, mCurrEntry, mCTFTree->GetEntries(), mCTFFile->GetName(), mTimer.CpuTime() - cput);

  bool moreToProcess = (++mCurrEntry < mCTFTree->GetEntries());
  if (!moreToProcess) { // this file is done, check if there are other files
    mCTFTree.reset();
    mCTFFile->Close();
    mCTFFile.reset();
    moreToProcess = true;
    if (++mNextToProcess >= mInput.size()) {
      if (++mLoopsCounter >= mLoops) {
        moreToProcess = false;
      } else {
        mNextToProcess = 0;
        LOG(INFO) << "Starting new loop " << mNextToProcess << " of " << mLoops;
      }
    }
  }

  mCTFCounter++;

  if (!moreToProcess) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    LOGP(INFO, "CTF reading total timing: Cpu: {:.3f} Real: {:.3f} s for {} TFs in {} loops",
         mTimer.CpuTime(), mTimer.RealTime(), mCTFCounter, mLoops);
  }
}

///_______________________________________
DataProcessorSpec getCTFReaderSpec(DetID::mask_t dets, const std::string& inp, int loop, int delayMUS)
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
    AlgorithmSpec{adaptFromTask<CTFReaderSpec>(dets, inp, loop, delayMUS)},
    Options{{"input-dir", VariantType::String, "none", {"CTF input directory"}}}};
}

} // namespace ctf
} // namespace o2
