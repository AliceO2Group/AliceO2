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
#include "Framework/RawDeviceService.h"
#include "Framework/RateLimiter.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/FileFetcher.h"
#include "CommonUtils/IRFrameSelector.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CTFWorkflow/CTFReaderSpec.h"
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "CommonUtils/NameConf.h"
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
#include <fairmq/Device.h>

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
  bool processTF(ProcessingContext& pc);
  void checkTreeEntries();
  void stopReader();
  template <typename C>
  void processDetector(DetID det, const CTFHeader& ctfHeader, ProcessingContext& pc) const;
  void setMessageHeader(ProcessingContext& pc, const CTFHeader& ctfHeader, const std::string& lbl, unsigned subspec) const; // keep just for the reference
  void tryToFixCTFHeader(CTFHeader& ctfHeader) const;
  CTFReaderInp mInput{};
  o2::utils::IRFrameSelector mIRFrameSelector; // optional IR frames selector
  std::unique_ptr<o2::utils::FileFetcher> mFileFetcher;
  std::unique_ptr<TFile> mCTFFile;
  std::unique_ptr<TTree> mCTFTree;
  bool mRunning = false;
  bool mUseLocalTFCounter = false;
  int mCTFCounter = 0;
  int mNFailedFiles = 0;
  int mFilesRead = 0;
  int mTFLength = 128;
  long mLastSendTime = 0L;
  long mCurrTreeEntry = 0L;
  long mImposeRunStartMS = 0L;
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
  LOGP(info, "CTFReader stops processing, {} files read, {} files failed", mFilesRead - mNFailedFiles, mNFailedFiles);
  LOGP(info, "CTF reading total timing: Cpu: {:.3f} Real: {:.3f} s for {} TFs in {} loops",
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
  mUseLocalTFCounter = ic.options().get<bool>("local-tf-counter");
  mImposeRunStartMS = ic.options().get<int64_t>("impose-run-start-timstamp");
  mRunning = true;
  mFileFetcher = std::make_unique<o2::utils::FileFetcher>(mInput.inpdata, mInput.tffileRegex, mInput.remoteRegex, mInput.copyCmd);
  mFileFetcher->setMaxFilesInQueue(mInput.maxFileCache);
  mFileFetcher->setMaxLoops(mInput.maxLoops);
  mFileFetcher->setFailThreshold(ic.options().get<float>("fetch-failure-threshold"));
  mFileFetcher->start();
  if (!mInput.fileIRFrames.empty()) {
    mIRFrameSelector.loadIRFrames(mInput.fileIRFrames);
    const auto& hbfu = o2::raw::HBFUtils::Instance();
    mTFLength = hbfu.nHBFPerTF;
    LOGP(info, "IRFrames will be selected from {}, assumed TF length: {} HBF", mInput.fileIRFrames, mTFLength);
  }
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
    LOG(error) << "Cannot process " << flname << ", reason: " << e.what();
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
  static bool initOnceDone = false;
  if (!initOnceDone) {
    mInput.tfRateLimit = std::stoi(pc.services().get<RawDeviceService>().device()->fConfig->GetValue<std::string>("timeframes-rate-limit"));
  }

  std::string tfFileName;
  if (mCTFCounter >= mInput.maxTFs || (!mInput.ctfIDs.empty() && mSelIDEntry >= mInput.ctfIDs.size())) { // done
    LOG(info) << "All CTFs from selected range were injected, stopping";
    mRunning = false;
  }

  while (mRunning) {
    if (mCTFTree) { // there is a tree open with multiple CTF
      if (mInput.ctfIDs.empty() || mInput.ctfIDs[mSelIDEntry] == mCTFCounter) { // no selection requested or matching CTF ID is found
        LOG(debug) << "TF " << mCTFCounter << " of " << mInput.maxTFs << " loop " << mFileFetcher->getNLoops();
        mSelIDEntry++;
        if (processTF(pc)) {
          break;
        }
      }
      // explict CTF ID selection list or IRFrame was provided and current entry is not selected
      LOGP(info, "Skipping CTF#{} ({} of {} in {})", mCTFCounter, mCurrTreeEntry, mCTFTree->GetEntries(), mCTFFile->GetName());
      checkTreeEntries();
      mCTFCounter++;
      continue;
    }
    //
    tfFileName = mFileFetcher->getNextFileInQueue();
    if (tfFileName.empty()) {
      if (!mFileFetcher->isRunning()) { // nothing expected in the queue
        mRunning = false;
        break;
      }
      pc.services().get<RawDeviceService>().waitFor(5);
      continue;
    }
    LOG(info) << "Reading CTF input " << ' ' << tfFileName;
    openCTFFile(tfFileName);
  }

  if (!mRunning) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    stopReader();
  }
}

///_______________________________________
bool CTFReaderSpec::processTF(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  static RateLimiter limiter;
  limiter.check(pc, mInput.tfRateLimit, mInput.minSHM);

  CTFHeader ctfHeader;
  if (!readFromTree(*(mCTFTree.get()), "CTFHeader", ctfHeader, mCurrTreeEntry)) {
    throw std::runtime_error("did not find CTFHeader");
  }
  if (mImposeRunStartMS > 0) {
    ctfHeader.creationTime = mImposeRunStartMS + ctfHeader.firstTForbit * o2::constants::lhc::LHCOrbitMUS * 1e-3;
  }
  if (ctfHeader.creationTime == 0) { // try to repair header with ad hoc data
    tryToFixCTFHeader(ctfHeader);
  }

  if (mUseLocalTFCounter) {
    ctfHeader.tfCounter = mCTFCounter;
  }

  LOG(info) << ctfHeader;

  auto& timingInfo = pc.services().get<o2::framework::TimingInfo>();
  timingInfo.firstTForbit = ctfHeader.firstTForbit;
  timingInfo.creation = ctfHeader.creationTime;
  timingInfo.tfCounter = ctfHeader.tfCounter;
  timingInfo.runNumber = ctfHeader.run;

  if (mIRFrameSelector.isSet()) {
    o2::InteractionRecord ir0(0, timingInfo.firstTForbit);
    // we cannot have GRPECS via DPL CCDB fetcher in the CTFReader, so we use mTFLength extracted from the HBFUtils
    o2::InteractionRecord ir1(o2::constants::lhc::LHCMaxBunches - 1, timingInfo.firstTForbit < 0xffffffff - (mTFLength - 1) ? timingInfo.firstTForbit + (mTFLength - 1) : 0xffffffff);
    auto irSpan = mIRFrameSelector.getMatchingFrames({ir0, ir1});
    if (irSpan.size() == 0 && mInput.skipSkimmedOutTF) {
      LOGP(info, "Skimming did not define any selection for TF [{}] : [{}]", ir0.asString(), ir1.asString());
      return false;
    } else {
      LOGP(info, "{} IR-Frames are selected for TF [{}] : [{}]", irSpan.size(), ir0.asString(), ir1.asString());
    }
    auto outVec = pc.outputs().make<std::vector<o2::dataformats::IRFrame>>(OutputRef{"selIRFrames"}, irSpan.begin(), irSpan.end());
  }

  // send CTF Header
  pc.outputs().snapshot({"header", mInput.subspec}, ctfHeader);

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
  if (!mInput.sup0xccdb) {
    auto& stfDist = pc.outputs().make<o2::header::STFHeader>(OutputRef{"TFDist", 0xccdb});
    stfDist.id = uint64_t(mCurrTreeEntry);
    stfDist.firstOrbit = ctfHeader.firstTForbit;
    stfDist.runNumber = uint32_t(ctfHeader.run);
  }

  auto entryStr = fmt::format("({} of {} in {})", mCurrTreeEntry, mCTFTree->GetEntries(), mCTFFile->GetName());
  checkTreeEntries();
  mTimer.Stop();

  // do we need to way to respect the delay ?
  long tNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  if (mCTFCounter) {
    auto tDiff = tNow - mLastSendTime;
    if (tDiff < mInput.delay_us) {
      pc.services().get<RawDeviceService>().waitFor((mInput.delay_us - tDiff) / 1000); // respect requested delay before sending
    }
  }
  tNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  LOGP(info, "Read CTF {} {} in {:.3f} s, {:.4f} s elapsed from previous CTF", mCTFCounter, entryStr, mTimer.CpuTime() - cput, mCTFCounter ? 1e-6 * (tNow - mLastSendTime) : 0.);
  mLastSendTime = tNow;
  mCTFCounter++;
  return true;
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
void CTFReaderSpec::setMessageHeader(ProcessingContext& pc, const CTFHeader& ctfHeader, const std::string& lbl, unsigned subspec) const
{
  auto* stack = pc.outputs().findMessageHeaderStack(OutputRef{lbl, subspec});
  if (!stack) {
    throw std::runtime_error(fmt::format("failed to find output message header stack for {}", lbl));
  }
  auto dh = const_cast<o2::header::DataHeader*>(o2::header::get<o2::header::DataHeader*>(stack));
  dh->firstTForbit = ctfHeader.firstTForbit;
  dh->tfCounter = ctfHeader.tfCounter;
  dh->runNumber = uint32_t(ctfHeader.run);
  auto dph = const_cast<o2::framework::DataProcessingHeader*>(o2::header::get<o2::framework::DataProcessingHeader*>(stack));
  dph->creation = ctfHeader.creationTime;
}

///_______________________________________
template <typename C>
void CTFReaderSpec::processDetector(DetID det, const CTFHeader& ctfHeader, ProcessingContext& pc) const
{
  if (mInput.detMask[det]) {
    const auto lbl = det.getName();
    auto& bufVec = pc.outputs().make<std::vector<o2::ctf::BufferType>>({lbl, mInput.subspec}, ctfHeader.detectors[det] ? sizeof(C) : 0);
    if (ctfHeader.detectors[det]) {
      C::readFromTree(bufVec, *(mCTFTree.get()), lbl, mCurrTreeEntry);
    } else if (!mInput.allowMissingDetectors) {
      throw std::runtime_error(fmt::format("Requested detector {} is missing in the CTF", lbl));
    }
    //    setMessageHeader(pc, ctfHeader, lbl);
  }
}

///_______________________________________
void CTFReaderSpec::tryToFixCTFHeader(CTFHeader& ctfHeader) const
{
  // HACK: fix CTFHeader for the pilot beam runs, where the TF creation time was not recorded
  struct RunStartData {
    uint32_t run = 0;
    uint32_t firstTForbit = 0;
    uint64_t tstampMS0 = 0;
  };
  const std::vector<RunStartData> tf0Data{
    {505207, 133875, 1635322620830},
    {505217, 14225007, 1635328375618},
    {505278, 1349340, 1635376882079},
    {505285, 1488862, 1635378517248},
    {505303, 2615411, 1635392586314},
    {505397, 5093945, 1635454778123},
    {505404, 19196217, 1635456032855},
    {505405, 28537913, 1635456862913},
    {505406, 41107641, 1635457980628},
    {505413, 452530, 1635460562613},
    {505440, 13320708, 1635472436927},
    {505443, 26546564, 1635473613239},
    {505446, 177711, 1635477270241},
    {505548, 88037114, 1635544414050},
    {505582, 295044346, 1635562822389},
    {505600, 417241082, 1635573688564},
    {505623, 10445984, 1635621310460},
    {505629, 126979, 1635623289756},
    {505637, 338969, 1635630909893},
    {505645, 188222, 1635634560881},
    {505658, 81044, 1635645404694},
    {505669, 328291, 1635657807147},
    {505673, 30988, 1635659148972},
    {505713, 620506, 1635725054798},
    {505720, 5359903, 1635730673978}};
  if (ctfHeader.run >= tf0Data.front().run && ctfHeader.run <= tf0Data.back().run) {
    for (const auto& tf0 : tf0Data) {
      if (ctfHeader.run == tf0.run) {
        ctfHeader.creationTime = tf0.tstampMS0;
        int64_t offset = std::ceil((ctfHeader.firstTForbit - tf0.firstTForbit) * o2::constants::lhc::LHCOrbitMUS * 1e-3);
        ctfHeader.creationTime += offset > 0 ? offset : 0;
        break;
      }
    }
  }
}

///_______________________________________
DataProcessorSpec getCTFReaderSpec(const CTFReaderInp& inp)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  std::vector<ConfigParamSpec> options;

  outputs.emplace_back(OutputLabel{"header"}, "CTF", "HEADER", inp.subspec, Lifetime::Timeframe);
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (inp.detMask[id]) {
      DetID det(id);
      outputs.emplace_back(OutputLabel{det.getName()}, det.getDataOrigin(), "CTFDATA", inp.subspec, Lifetime::Timeframe);
    }
  }
  if (!inp.fileIRFrames.empty()) {
    outputs.emplace_back(OutputLabel{"selIRFrames"}, "CTF", "SELIRFRAMES", 0, Lifetime::Timeframe);
  }
  if (!inp.sup0xccdb) {
    outputs.emplace_back(OutputSpec{{"TFDist"}, o2::header::gDataOriginFLP, o2::header::gDataDescriptionDISTSTF, 0xccdb});
  }

  options.emplace_back(ConfigParamSpec{"select-ctf-ids", VariantType::String, "", {"comma-separated list CTF IDs to inject (from cumulative counter of CTFs seen)"}});
  options.emplace_back(ConfigParamSpec{"impose-run-start-timstamp", VariantType::Int64, 0L, {"impose run start time stamp (ms), ignored if 0"}});
  options.emplace_back(ConfigParamSpec{"local-tf-counter", VariantType::Bool, false, {"reassign header.tfCounter from local TF counter"}});
  options.emplace_back(ConfigParamSpec{"fetch-failure-threshold", VariantType::Float, 0.f, {"Fatil if too many failures( >0: fraction, <0: abs number, 0: no threshold)"}});
  if (!inp.metricChannel.empty()) {
    options.emplace_back(ConfigParamSpec{"channel-config", VariantType::String, inp.metricChannel, {"Out-of-band channel config for TF throttling"}});
  }

  return DataProcessorSpec{
    "ctf-reader",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CTFReaderSpec>(inp)},
    options};
}

} // namespace ctf
} // namespace o2
