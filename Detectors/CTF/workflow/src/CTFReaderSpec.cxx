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
#include "Headers/STFHeader.h"
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
  template <typename C>
  void processDetector(DetID det, const CTFHeader& ctfHeader, ProcessingContext& pc) const;
  void setFirstTFOrbit(const CTFHeader& ctfHeader, const std::string& lbl, ProcessingContext& pc) const;
  CTFReaderInp mInput{};
  std::unique_ptr<o2::utils::FileFetcher> mFileFetcher;
  std::unique_ptr<TFile> mCTFFile;
  std::unique_ptr<TTree> mCTFTree;
  bool mRunning = false;
  int mCTFCounter = 0;
  int mNFailedFiles = 0;
  int mFilesRead = 0;
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
  LOGP(INFO, "CTFReader stops processing, {} files read, {} files failed", mFilesRead - mNFailedFiles, mNFailedFiles);
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
  try {
    mFilesRead++;
    mCTFFile.reset(TFile::Open(flname.c_str()));
    if (!mCTFFile || !mCTFFile->IsOpen() || mCTFFile->IsZombie()) {
      throw std::runtime_error("failed to open CTF file");
    }
    mCTFTree.reset((TTree*)mCTFFile->Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    if (!mCTFTree) {
      throw std::runtime_error("failed to load CTF tree from");
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Cannot process " << flname << ", reason: " << e.what();
    mCTFTree.reset();
    mCTFFile.reset();
    mNFailedFiles++;
    if (mFileFetcher) {
      mFileFetcher->popFromQueue(mInput.maxLoops < 1);
    }
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

  // send CTF Header
  pc.outputs().snapshot({"header"}, ctfHeader);
  setFirstTFOrbit(ctfHeader, "header", pc);

  processDetector<o2::itsmft::CTF>(DetID::ITS, ctfHeader, pc);
  processDetector<o2::itsmft::CTF>(DetID::MFT, ctfHeader, pc);
  processDetector<o2::emcal::CTF>(DetID::EMC, ctfHeader, pc);
  processDetector<o2::hmpid::CTF>(DetID::HMP, ctfHeader, pc);
  processDetector<o2::phos::CTF>(DetID::PHS, ctfHeader, pc);
  processDetector<o2::tpc::CTF>(DetID::TPC, ctfHeader, pc);
  processDetector<o2::trd::CTF>(DetID::TRD, ctfHeader, pc);
  processDetector<o2::ft0::CTF>(DetID::FT0, ctfHeader, pc);
  processDetector<o2::fv0::CTF>(DetID::FV0, ctfHeader, pc);
  processDetector<o2::fdd::CTF>(DetID::FDD, ctfHeader, pc);
  processDetector<o2::tof::CTF>(DetID::TOF, ctfHeader, pc);
  processDetector<o2::mid::CTF>(DetID::MID, ctfHeader, pc);
  processDetector<o2::mch::CTF>(DetID::MCH, ctfHeader, pc);
  processDetector<o2::cpv::CTF>(DetID::CPV, ctfHeader, pc);
  processDetector<o2::zdc::CTF>(DetID::ZDC, ctfHeader, pc);
  processDetector<o2::ctp::CTF>(DetID::CTP, ctfHeader, pc);

  // send sTF acknowledge message
  {
    auto& stfDist = pc.outputs().make<o2::header::STFHeader>({"STFDist"});
    stfDist.id = uint64_t(mCurrTreeEntry);
    stfDist.firstOrbit = ctfHeader.firstTForbit;
    stfDist.runNumber = uint32_t(ctfHeader.run);
    setFirstTFOrbit(ctfHeader, "STFDist", pc);
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
void CTFReaderSpec::setFirstTFOrbit(const CTFHeader& ctfHeader, const std::string& lbl, ProcessingContext& pc) const
{
  auto* hd = pc.outputs().findMessageHeader({lbl});
  if (!hd) {
    throw std::runtime_error(fmt::format("failed to find output message header for {}", lbl));
  }
  hd->firstTForbit = ctfHeader.firstTForbit;
  hd->tfCounter = mCTFCounter;
  hd->runNumber = uint32_t(ctfHeader.run);
}

///_______________________________________
template <typename C>
void CTFReaderSpec::processDetector(DetID det, const CTFHeader& ctfHeader, ProcessingContext& pc) const
{
  if (mInput.detMask[det]) {
    const auto lbl = det.getName();
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({lbl}, sizeof(C));
    if (ctfHeader.detectors[det]) {
      C::readFromTree(bufVec, *(mCTFTree.get()), lbl, mCurrTreeEntry);
    } else if (!mInput.allowMissingDetectors) {
      throw std::runtime_error(fmt::format("Requested detector {} is missing in the CTF", lbl));
    }
    setFirstTFOrbit(ctfHeader, lbl, pc);
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
  outputs.emplace_back(OutputSpec{{"STFDist"}, o2::header::gDataOriginFLP, o2::header::gDataDescriptionDISTSTF, 0});

  return DataProcessorSpec{
    "ctf-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<CTFReaderSpec>(inp)},
    Options{{"select-ctf-ids", VariantType::String, "", {"comma-separated list CTF IDs to inject (from cumulative counter of CTFs seen)"}}}};
}

} // namespace ctf
} // namespace o2
