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

/// @file   CTFReaderSpec.cxx

#include <vector>
#include <TFile.h>
#include <TTree.h>

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/FileFetcher.h"
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
#include "DataFormatsCTP/CTF.h"
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
  CTFReaderSpec(const CTFReaderInp& inp);
  ~CTFReaderSpec() override;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void openCTFFile(const std::string& flname);
  void processTF(ProcessingContext& pc);
  void checkTreeEntries();
  void stopReader();
  CTFReaderInp mInput{};
  std::unique_ptr<o2::utils::FileFetcher> mFileFetcher;
  std::unique_ptr<TFile> mCTFFile;
  std::unique_ptr<TTree> mCTFTree;
  bool mRunning = false;
  int mCTFCounter = 0;
  long mLastSendTime = 0L;
  long mCurrTreeEntry = 0;
  size_t mSelIDEntry = 0; // next CTFID to select from the mInput.ctfIDs (if non-empty)
  TStopwatch mTimer;
};

///_______________________________________
CTFReaderSpec::CTFReaderSpec(const CTFReaderInp& inp) : mInput(inp)
{
  mTimer.Stop();
  mTimer.Reset();
}

///_______________________________________
CTFReaderSpec::~CTFReaderSpec()
{
  stopReader();
}

///_______________________________________
void CTFReaderSpec::stopReader()
{
  if (!mFileFetcher) {
    return;
  }
  LOG(INFO) << "CTFReader stops processing";
  LOGP(INFO, "CTF reading total timing: Cpu: {:.3f} Real: {:.3f} s for {} TFs in {} loops",
       mTimer.CpuTime(), mTimer.RealTime(), mCTFCounter, mFileFetcher->getNLoops());
  mRunning = false;
  mFileFetcher->stop();
  mFileFetcher.reset();
  mCTFTree.reset();
  if (mCTFFile) {
    mCTFFile->Close();
  }
  mCTFFile.reset();
}

///_______________________________________
void CTFReaderSpec::init(InitContext& ic)
{
  mInput.ctfIDs = o2::RangeTokenizer::tokenize<int>(ic.options().get<std::string>("select-ctf-ids"));
  mRunning = true;
  mFileFetcher = std::make_unique<o2::utils::FileFetcher>(mInput.inpdata, mInput.tffileRegex, mInput.remoteRegex, mInput.copyCmd);
  mFileFetcher->setMaxFilesInQueue(mInput.maxFileCache);
  mFileFetcher->setMaxLoops(mInput.maxLoops);
  mFileFetcher->start();
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
  mCurrTreeEntry = 0;
}

///_______________________________________
void CTFReaderSpec::run(ProcessingContext& pc)
{
  std::string tfFileName;
  if (mCTFCounter >= mInput.maxTFs || (!mInput.ctfIDs.empty() && mSelIDEntry >= mInput.ctfIDs.size())) { // done
    LOG(INFO) << "All CTFs from selected range were injected, stopping";
    mRunning = false;
  }

  while (mRunning) {
    if (mCTFTree) { // there is a tree open with multiple CTF
      if (mInput.ctfIDs.empty() || mInput.ctfIDs[mSelIDEntry] == mCTFCounter) { // no selection requested or matching CTF ID is found
        LOG(DEBUG) << "TF " << mCTFCounter << " of " << mInput.maxTFs << " loop " << mFileFetcher->getNLoops();
        mSelIDEntry++;
        processTF(pc);
        break;
      } else { // explict CTF ID selection list was provided and current entry is not selected
        LOGP(INFO, "Skipping CTF${} ({} of {} in {})", mCTFCounter, mCurrTreeEntry, mCTFTree->GetEntries(), mCTFFile->GetName());
        checkTreeEntries();
        mCTFCounter++;
        continue;
      }
    }
    //
    tfFileName = mFileFetcher->getNextFileInQueue();
    if (tfFileName.empty()) {
      if (!mFileFetcher->isRunning()) { // nothing expected in the queue
        mRunning = false;
        break;
      }
      usleep(5000); // wait 5ms for the files cache to be filled
      continue;
    }
    LOG(INFO) << "Reading CTF input " << ' ' << tfFileName;
    openCTFFile(tfFileName);
  }

  if (!mRunning) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    stopReader();
  }
}

///_______________________________________
void CTFReaderSpec::processTF(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  CTFHeader ctfHeader;
  if (!readFromTree(*(mCTFTree.get()), "CTFHeader", ctfHeader, mCurrTreeEntry)) {
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

  DetID::mask_t detsTF = mInput.detMask & ctfHeader.detectors;
  DetID det;

  det = DetID::ITS;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::itsmft::CTF));
    o2::itsmft::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::MFT;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::itsmft::CTF));
    o2::itsmft::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::TPC;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::tpc::CTF));
    o2::tpc::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::TRD;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::trd::CTF));
    o2::trd::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::FT0;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::ft0::CTF));
    o2::ft0::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::FV0;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::fv0::CTF));
    o2::fv0::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::FDD;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::fdd::CTF));
    o2::fdd::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::TOF;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::tof::CTF));
    o2::tof::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::MID;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::mid::CTF));
    o2::mid::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::MCH;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::mch::CTF));
    o2::mch::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::EMC;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::emcal::CTF));
    o2::emcal::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::PHS;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::phos::CTF));
    o2::phos::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::CPV;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::cpv::CTF));
    o2::cpv::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::ZDC;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::zdc::CTF));
    o2::zdc::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::HMP;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::hmpid::CTF));
    o2::hmpid::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  det = DetID::CTP;
  if (detsTF[det]) {
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({det.getName()}, sizeof(o2::ctp::CTF));
    o2::ctp::CTF::readFromTree(bufVec, *(mCTFTree.get()), det.getName(), mCurrTreeEntry);
    setFirstTFOrbit(det.getName());
  }

  auto entryStr = fmt::format("({} of {} in {})", mCurrTreeEntry, mCTFTree->GetEntries(), mCTFFile->GetName());
  checkTreeEntries();
  mTimer.Stop();
  // do we need to way to respect the delay ?
  long tNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  auto tDiff = tNow - mLastSendTime;
  if (mCTFCounter) {
    if (tDiff < mInput.delay_us) {
      usleep(mInput.delay_us - tDiff); // respect requested delay before sending
    }
  } else {
    mLastSendTime = tNow;
  }
  tNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  LOGP(INFO, "Read CTF#{} {} in {:.3f} s, {:.4f} s elapsed from previous CTF", mCTFCounter, entryStr, mTimer.CpuTime() - cput, 1e-6 * (tNow - mLastSendTime));
  mLastSendTime = tNow;
  mCTFCounter++;
}

///_______________________________________
void CTFReaderSpec::checkTreeEntries()
{
  // check if the tree has entries left, if needed, close current tree/file
  if (++mCurrTreeEntry >= mCTFTree->GetEntries()) { // this file is done, check if there are other files
    mCTFTree.reset();
    mCTFFile->Close();
    mCTFFile.reset();
    if (mFileFetcher) {
      mFileFetcher->popFromQueue(mInput.maxLoops < 1);
    }
  }
}

///_______________________________________
DataProcessorSpec getCTFReaderSpec(const CTFReaderInp& inp)
{
  std::vector<OutputSpec> outputs;

  outputs.emplace_back(OutputLabel{"header"}, "CTF", "HEADER", 0, Lifetime::Timeframe);
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (inp.detMask[id]) {
      DetID det(id);
      outputs.emplace_back(OutputLabel{det.getName()}, det.getDataOrigin(), "CTFDATA", 0, Lifetime::Timeframe);
    }
  }
  return DataProcessorSpec{
    "ctf-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<CTFReaderSpec>(inp)},
    Options{{"select-ctf-ids", VariantType::String, "", {"comma-separated list CTF IDs to inject (from cumulative counter of CTFs seen)"}}}};
}

} // namespace ctf
} // namespace o2
