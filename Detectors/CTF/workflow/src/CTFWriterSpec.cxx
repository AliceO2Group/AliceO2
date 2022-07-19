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

/// @file   CTFWriterSpec.cxx

#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/CommonServices.h"
#include "Framework/DataTakingContext.h"
#include "Framework/TimingInfo.h"
#include <fairmq/Device.h>

#include "DataFormatsParameters/GRPECSObject.h"
#include "CTFWorkflow/CTFWriterSpec.h"
#include "DetectorsCommonDataFormats/CTFHeader.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/FileSystemUtils.h"
#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "CommonUtils/StringUtils.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsTRD/CTF.h"
#include "DataFormatsHMP/CTF.h"
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
#include "DataFormatsCTP/CTF.h"
#include "rANS/rans.h"
#include <vector>
#include <stdexcept>
#include <array>
#include <TStopwatch.h>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>
#include <filesystem>
#include <ctime>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <regex>

using namespace o2::framework;

namespace o2
{
namespace ctf
{

template <typename T>
size_t appendToTree(TTree& tree, const std::string brname, T& ptr)
{
  size_t s = 0;
  auto* br = tree.GetBranch(brname.c_str());
  auto* pptr = &ptr;
  if (br) {
    br->SetAddress(&pptr);
  } else {
    br = tree.Branch(brname.c_str(), &pptr);
  }
  int res = br->Fill();
  if (res < 0) {
    throw std::runtime_error(fmt::format("Failed to fill CTF branch {}", brname));
  }
  s += res;
  br->ResetAddress();
  return s;
}

using DetID = o2::detectors::DetID;
using FTrans = o2::rans::FrequencyTable;

class CTFWriterSpec : public o2::framework::Task
{
 public:
  CTFWriterSpec() = delete;
  CTFWriterSpec(DetID::mask_t dm, uint64_t r, const std::string& outType, int verbosity, int reportInterval);
  ~CTFWriterSpec() final { finalize(); }
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final { finalize(); }
  void stop() final { finalize(); }
  bool isPresent(DetID id) const { return mDets[id]; }

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  template <typename C>
  size_t processDet(o2::framework::ProcessingContext& pc, DetID det, CTFHeader& header, TTree* tree);
  template <typename C>
  void storeDictionary(DetID det, CTFHeader& header);
  void storeDictionaries();
  void closeTFTreeAndFile();
  void prepareTFTreeAndFile();
  size_t estimateCTFSize(ProcessingContext& pc);
  size_t getAvailableDiskSpace(const std::string& path, int level);
  void createLockFile(int level);
  void removeLockFile();
  void finalize();

  DetID::mask_t mDets; // detectors
  bool mFinalized = false;
  bool mWriteCTF = true;
  bool mCreateDict = false;
  bool mCreateRunEnvDir = true;
  bool mStoreMetaFile = false;
  bool mRejectCurrentTF = false;
  int mReportInterval = -1;
  int mVerbosity = 0;
  int mSaveDictAfter = 0;          // if positive and mWriteCTF==true, save dictionary after each mSaveDictAfter TFs processed
  uint32_t mPrevDictTimeStamp = 0; // timestamp of the previously stored dictionary
  uint32_t mDictTimeStamp = 0;     // timestamp of the currently stored dictionary
  size_t mMinSize = 0;               // if > 0, accumulate CTFs in the same tree until the total size exceeds this minimum
  size_t mMaxSize = 0;               // if > MinSize, and accumulated size will exceed this value, stop accumulation (even if mMinSize is not reached)
  size_t mChkSize = 0;               // if > 0 and fallback storage provided, reserve this size per CTF file in production on primary storage
  size_t mAccCTFSize = 0;            // so far accumulated size (if any)
  size_t mCurrCTFSize = 0;           // size of currently processed CTF
  size_t mNCTF = 0;                  // total number of CTFs written
  size_t mNCTFPrevDict = 0;          // total number of CTFs used for previous dictionary version
  size_t mNAccCTF = 0;               // total number of CTFs accumulated in the current file
  size_t mCTFAutoSave = 0;           // if > 0, autosave after so many TFs
  size_t mNCTFFiles = 0;             // total number of CTF files written
  int mMaxCTFPerFile = 0;            // max CTFs per files to store
  int mRejRate = 0;                  // CTF rejection rule (>0: percentage to reject randomly, <0: reject if timeslice%|value|!=0)
  std::vector<uint32_t> mTFOrbits{}; // 1st orbits of TF accumulated in current file
  o2::framework::DataTakingContext mDataTakingContext{};
  o2::framework::TimingInfo mTimingInfo{};
  std::string mOutputType{}; // RS FIXME once global/local options clash is solved, --output-type will become device option
  std::string mDictDir{};
  std::string mCTFDir{};
  std::string mHostName{};
  std::string mCTFDirFallBack = "/dev/null";
  std::string mCTFMetaFileDir = "/dev/null";
  std::string mCurrentCTFFileName{};
  std::string mCurrentCTFFileNameFull{};
  std::string mSizeReport{};
  std::string mMetaDataType{};
  const std::string LOCKFileDir = "/tmp/ctf-writer-locks";
  std::string mLockFileName{};
  int mLockFD = -1;
  std::unique_ptr<TFile> mCTFFileOut;
  std::unique_ptr<TTree> mCTFTreeOut;

  std::unique_ptr<TFile> mDictFileOut; // file to store dictionary
  std::unique_ptr<TTree> mDictTreeOut; // tree to store dictionary

  // For the external dictionary creation we accumulate for each detector the frequency tables of its each block
  // After accumulation over multiple TFs we store the dictionaries data in the standard CTF format of this detector,
  // i.e. EncodedBlock stored in a tree, BUT with dictionary data only added to each block.
  // The metadata of the block (min,max) will be used for the consistency check at the decoding
  std::array<std::vector<FTrans>, DetID::nDetectors> mFreqsAccumulation;
  std::array<std::vector<o2::ctf::Metadata>, DetID::nDetectors> mFreqsMetaData;
  std::array<std::bitset<64>, DetID::nDetectors> mIsSaturatedFrequencyTable;
  std::array<std::shared_ptr<void>, DetID::nDetectors> mHeaders;
  TStopwatch mTimer;

  static const std::string TMPFileEnding;
};

const std::string CTFWriterSpec::TMPFileEnding{".part"};

//___________________________________________________________________
CTFWriterSpec::CTFWriterSpec(DetID::mask_t dm, uint64_t r, const std::string& outType, int verbosity, int reportInterval)
  : mDets(dm), mOutputType(outType), mReportInterval(reportInterval), mVerbosity(verbosity)
{
  std::for_each(mIsSaturatedFrequencyTable.begin(), mIsSaturatedFrequencyTable.end(), [](auto& bitset) { bitset.reset(); });
  mTimer.Stop();
  mTimer.Reset();
}

//___________________________________________________________________
void CTFWriterSpec::init(InitContext& ic)
{
  // auto outmode = ic.options().get<std::string>("output-type"); // RS FIXME once global/local options clash is solved, --output-type will become device option
  auto outmode = mOutputType;
  if (outmode == "ctf") {
    mWriteCTF = true;
    mCreateDict = false;
  } else if (outmode == "dict") {
    mWriteCTF = false;
    mCreateDict = true;
  } else if (outmode == "both") {
    mWriteCTF = true;
    mCreateDict = true;
  } else if (outmode == "none") {
    mWriteCTF = false;
    mCreateDict = false;
  } else {
    throw std::invalid_argument("Invalid output-type");
  }

  mSaveDictAfter = ic.options().get<int>("save-dict-after");
  mCTFAutoSave = ic.options().get<int>("save-ctf-after");
  mDictDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("ctf-dict-dir"));
  mCTFDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("output-dir"));
  mCTFDirFallBack = ic.options().get<std::string>("output-dir-alt");
  if (mCTFDirFallBack != "/dev/null") {
    mCTFDirFallBack = o2::utils::Str::rectifyDirectory(mCTFDirFallBack);
  }
  mCTFMetaFileDir = ic.options().get<std::string>("meta-output-dir");
  if (mCTFMetaFileDir != "/dev/null") {
    mCTFMetaFileDir = o2::utils::Str::rectifyDirectory(mCTFMetaFileDir);
    mStoreMetaFile = true;
  }
  mCreateRunEnvDir = !ic.options().get<bool>("ignore-partition-run-dir");
  mMinSize = ic.options().get<int64_t>("min-file-size");
  mMaxSize = ic.options().get<int64_t>("max-file-size");
  mMaxCTFPerFile = ic.options().get<int>("max-ctf-per-file");
  mRejRate = ic.options().get<int>("ctf-rejection");
  if (mRejRate > 0) {
    LOGP(info, "Will reject{} {}% of TFs", mRejRate < 100 ? " randomly" : "", mRejRate < 100 ? mRejRate : 100);
  } else if (mRejRate < -1) {
    LOGP(info, "Will reject all but each {}-th TF slice", -mRejRate);
  }

  if (mWriteCTF) {
    if (mMinSize > 0) {
      LOG(info) << "Multiple CTFs will be accumulated in the tree/file until its size exceeds " << mMinSize << " bytes";
      if (mMaxSize > mMinSize) {
        LOG(info) << "but does not exceed " << mMaxSize << " bytes";
      }
    }
  }
  mChkSize = std::max(size_t(mMinSize * 1.1), mMaxSize);
  o2::utils::createDirectoriesIfAbsent(LOCKFileDir);

  if (mCreateDict) { // make sure that there is no local dictonary
    std::string dictFileName = fmt::format("{}{}.root", mDictDir, o2::base::NameConf::CTFDICT);
    if (std::filesystem::exists(dictFileName)) {
      throw std::runtime_error(o2::utils::Str::concat_string("CTF dictionary creation is requested but ", dictFileName, " already exists, remove it!"));
    }
    o2::utils::createDirectoriesIfAbsent(mDictDir);
  }

  char hostname[_POSIX_HOST_NAME_MAX];
  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  mHostName = hostname;
  mHostName = mHostName.substr(0, mHostName.find('.'));
}

//___________________________________________________________________
void CTFWriterSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  using GRPECS = o2::parameters::GRPECSObject;
  if (!initOnceDone) {
    initOnceDone = true;
    mDataTakingContext = pc.services().get<DataTakingContext>();
    // determine the output type for the CTF metadata
    mMetaDataType = GRPECS::getRawDataPersistencyMode(mDataTakingContext.runType, mDataTakingContext.forcedRaw);
  }
  mTimingInfo = pc.services().get<o2::framework::TimingInfo>();
}

//___________________________________________________________________
// process data of particular detector
template <typename C>
size_t CTFWriterSpec::processDet(o2::framework::ProcessingContext& pc, DetID det, CTFHeader& header, TTree* tree)
{
  size_t sz = 0;
  if (!isPresent(det) || !pc.inputs().isValid(det.getName())) {
    mSizeReport += fmt::format(" {}:N/A", det.getName());
    return sz;
  }
  auto ctfBuffer = pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName());
  const auto ctfImage = C::getImage(ctfBuffer.data());
  ctfImage.print(o2::utils::Str::concat_string(det.getName(), ": "), mVerbosity);
  if (mWriteCTF && !mRejectCurrentTF) {
    sz = ctfImage.appendToTree(*tree, det.getName());
    header.detectors.set(det);
  } else {
    sz = ctfBuffer.size();
  }
  if (mCreateDict) {
    if (!mFreqsAccumulation[det].size()) {
      mFreqsAccumulation[det].resize(C::getNBlocks());
      mFreqsMetaData[det].resize(C::getNBlocks());
    }
    if (!mHeaders[det]) { // store 1st header
      mHeaders[det] = ctfImage.cloneHeader();
      auto& hb = *static_cast<o2::ctf::CTFDictHeader*>(mHeaders[det].get());
      hb.det = det;
    }
    for (int ib = 0; ib < C::getNBlocks(); ib++) {
      if (!mIsSaturatedFrequencyTable[det][ib]) {
        const auto& bl = ctfImage.getBlock(ib);
        if (bl.getNDict()) {
          auto freq = mFreqsAccumulation[det][ib];
          auto& mdSave = mFreqsMetaData[det][ib];
          const auto& md = ctfImage.getMetadata(ib);
          if ([&, this]() {
                try {
                  freq.addFrequencies(bl.getDict(), bl.getDict() + bl.getNDict(), md.min);
                } catch (const std::overflow_error& e) {
                  LOGP(warning, "unable to frequency table for {}, block {} due to overflow", det.getName(), ib);
                  mIsSaturatedFrequencyTable[det][ib] = true;
                  return false;
                }
                return true;
              }()) {
            auto newProbBits = static_cast<uint8_t>(o2::rans::computeRenormingPrecision(freq));
            mdSave = o2::ctf::Metadata{0, 0, md.messageWordSize, md.coderType, md.streamSize, newProbBits, md.opt, freq.getMinSymbol(), freq.getMaxSymbol(), static_cast<int32_t>(freq.size()), 0, 0};
            mFreqsAccumulation[det][ib] = std::move(freq);
          }
        }
      }
    }
  }
  mSizeReport += fmt::format(" {}:{}", det.getName(), fmt::group_digits(sz));
  return sz;
}

//___________________________________________________________________
// store dictionary of a particular detector
template <typename C>
void CTFWriterSpec::storeDictionary(DetID det, CTFHeader& header)
{
  // create vector whose data contains dictionary in CTF format (EncodedBlock)
  if (!isPresent(det) || !mFreqsAccumulation[det].size()) {
    return;
  }
  auto dictBlocks = C::createDictionaryBlocks(mFreqsAccumulation[det], mFreqsMetaData[det]);
  auto& h = C::get(dictBlocks.data())->getHeader();
  h = *reinterpret_cast<typename std::remove_reference<decltype(h)>::type*>(mHeaders[det].get());
  auto& hb = static_cast<o2::ctf::CTFDictHeader&>(h);
  hb = *static_cast<const o2::ctf::CTFDictHeader*>(mHeaders[det].get());
  hb.dictTimeStamp = mDictTimeStamp;

  auto getFileName = [this, det, &hb](bool curr) {
    return fmt::format("{}{}_{}_v{}.{}_{}_{}.root", this->mDictDir, o2::base::NameConf::CTFDICT, det.getName(), int(hb.majorVersion), int(hb.minorVersion),
                       curr ? this->mDictTimeStamp : this->mPrevDictTimeStamp, curr ? this->mNCTF : this->mNCTFPrevDict);
  };

  C::get(dictBlocks.data())->print(o2::utils::Str::concat_string("Storing dictionary for ", det.getName(), ": "));
  auto outName = getFileName(true);
  TFile flout(outName.c_str(), "recreate");
  flout.WriteObject(&dictBlocks, o2::base::NameConf::CCDBOBJECT.data());
  flout.WriteObject(&hb, fmt::format("ctf_dict_header_{}", det.getName()).c_str());
  flout.Close();
  LOGP(info, "Saved {} with {} TFs to {}", hb.asString(), mNCTF, outName);
  if (mPrevDictTimeStamp) {
    auto outNamePrev = getFileName(false);
    if (std::filesystem::exists(outNamePrev)) {
      std::filesystem::remove(outNamePrev);
      LOGP(info, "Removed previous dictionary version {}", outNamePrev);
    }
  }
  C::get(dictBlocks.data())->appendToTree(*mDictTreeOut.get(), det.getName()); // cast to EncodedBlock and attach to dictionaries tree
  header.detectors.set(det);
}

//___________________________________________________________________
size_t CTFWriterSpec::estimateCTFSize(ProcessingContext& pc)
{
  size_t s = 0;
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    DetID det(id);
    if (!isPresent(det) || !pc.inputs().isValid(det.getName())) {
      continue;
    }
    s += pc.inputs().get<gsl::span<o2::ctf::BufferType>>(det.getName()).size();
  }
  return s;
}

//___________________________________________________________________
void CTFWriterSpec::run(ProcessingContext& pc)
{
  const std::string NAStr = "NA";
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  updateTimeDependentParams(pc);
  mRejectCurrentTF = (mRejRate > 0 && int(gRandom->Rndm() * 100) < mRejRate) || (mRejRate < -1 && mTimingInfo.timeslice % (-mRejRate));
  mCurrCTFSize = estimateCTFSize(pc);
  if (mWriteCTF && !mRejectCurrentTF) {
    prepareTFTreeAndFile();
  }
  // create header
  CTFHeader header{mTimingInfo.runNumber, mTimingInfo.creation, mTimingInfo.firstTForbit, mTimingInfo.tfCounter};
  size_t szCTF = 0;
  mSizeReport = "";
  szCTF += processDet<o2::itsmft::CTF>(pc, DetID::ITS, header, mCTFTreeOut.get());
  szCTF += processDet<o2::tpc::CTF>(pc, DetID::TPC, header, mCTFTreeOut.get());
  szCTF += processDet<o2::trd::CTF>(pc, DetID::TRD, header, mCTFTreeOut.get());
  szCTF += processDet<o2::tof::CTF>(pc, DetID::TOF, header, mCTFTreeOut.get());
  szCTF += processDet<o2::phos::CTF>(pc, DetID::PHS, header, mCTFTreeOut.get());
  szCTF += processDet<o2::cpv::CTF>(pc, DetID::CPV, header, mCTFTreeOut.get());
  szCTF += processDet<o2::emcal::CTF>(pc, DetID::EMC, header, mCTFTreeOut.get());
  szCTF += processDet<o2::hmpid::CTF>(pc, DetID::HMP, header, mCTFTreeOut.get());
  szCTF += processDet<o2::itsmft::CTF>(pc, DetID::MFT, header, mCTFTreeOut.get());
  szCTF += processDet<o2::mch::CTF>(pc, DetID::MCH, header, mCTFTreeOut.get());
  szCTF += processDet<o2::mid::CTF>(pc, DetID::MID, header, mCTFTreeOut.get());
  szCTF += processDet<o2::zdc::CTF>(pc, DetID::ZDC, header, mCTFTreeOut.get());
  szCTF += processDet<o2::ft0::CTF>(pc, DetID::FT0, header, mCTFTreeOut.get());
  szCTF += processDet<o2::fv0::CTF>(pc, DetID::FV0, header, mCTFTreeOut.get());
  szCTF += processDet<o2::fdd::CTF>(pc, DetID::FDD, header, mCTFTreeOut.get());
  szCTF += processDet<o2::ctp::CTF>(pc, DetID::CTP, header, mCTFTreeOut.get());
  if (mReportInterval > 0 && (mTimingInfo.tfCounter % mReportInterval) == 0) {
    LOGP(important, "CTF {} size report:{}", mTimingInfo.tfCounter, mSizeReport);
  }

  mTimer.Stop();

  if (mWriteCTF && !mRejectCurrentTF) {
    szCTF += appendToTree(*mCTFTreeOut.get(), "CTFHeader", header);
    mAccCTFSize += szCTF;
    mCTFTreeOut->SetEntries(++mNAccCTF);
    mTFOrbits.push_back(mTimingInfo.firstTForbit);
    LOG(info) << "TF#" << mNCTF << ": wrote CTF{" << header << "} of size " << szCTF << " to " << mCurrentCTFFileNameFull << " in " << mTimer.CpuTime() - cput << " s";
    if (mNAccCTF > 1) {
      LOG(info) << "Current CTF tree has " << mNAccCTF << " entries with total size of " << mAccCTFSize << " bytes";
    }
    if (mLockFD != -1) {
      lseek(mLockFD, 0, SEEK_SET);
      auto nwr = write(mLockFD, &mAccCTFSize, sizeof(size_t));
      if (nwr != sizeof(size_t)) {
        LOG(error) << "Failed to write current CTF size " << mAccCTFSize << " to lock file, bytes written: " << nwr;
      }
    }

    if (mAccCTFSize >= mMinSize || (mMaxCTFPerFile > 0 && mNAccCTF >= mMaxCTFPerFile)) {
      closeTFTreeAndFile();
    } else if (mCTFAutoSave > 0 && mNAccCTF % mCTFAutoSave == 0) {
      mCTFTreeOut->AutoSave("override");
    }
  } else {
    LOG(info) << "TF#" << mNCTF << " {" << header << "} CTF writing is disabled, size was " << szCTF << " bytes";
  }

  mNCTF++;
  if (mCreateDict && mSaveDictAfter > 0 && (mNCTF % mSaveDictAfter) == 0) {
    storeDictionaries();
  }
}

//___________________________________________________________________
void CTFWriterSpec::finalize()
{
  if (mFinalized) {
    return;
  }
  if (mCreateDict) {
    storeDictionaries();
  }
  if (mWriteCTF) {
    closeTFTreeAndFile();
  }
  LOGF(info, "CTF writing total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  mFinalized = true;
}

//___________________________________________________________________
void CTFWriterSpec::prepareTFTreeAndFile()
{
  if (!mWriteCTF) {
    return;
  }
  bool needToOpen = false;
  if (!mCTFTreeOut) {
    needToOpen = true;
  } else {
    if ((mAccCTFSize >= mMinSize) ||                                                         // min size exceeded, may close the file.
        (mAccCTFSize && mMaxSize > mMinSize && ((mAccCTFSize + mCurrCTFSize) > mMaxSize))) { // this is not the 1st CTF in the file and the new size will exceed allowed max
      needToOpen = true;
    } else {
      LOGP(info, "Will add new CTF of estimated size {} to existing file of size {}", mCurrCTFSize, mAccCTFSize);
    }
  }
  if (needToOpen) {
    closeTFTreeAndFile();
    auto ctfDir = mCTFDir.empty() ? o2::utils::Str::rectifyDirectory("./") : mCTFDir;
    if (mChkSize > 0 && (mCTFDirFallBack != "/dev/null")) {
      createLockFile(0);
      auto sz = getAvailableDiskSpace(ctfDir, 0); // check main storage
      if (sz < mChkSize) {
        removeLockFile();
        LOG(warning) << "Primary CTF output device has available size " << sz << " while " << mChkSize << " is requested: will write on secondary one";
        ctfDir = mCTFDirFallBack;
      }
    }
    if (mCreateRunEnvDir && !mDataTakingContext.envId.empty() && (mDataTakingContext.envId != o2::framework::DataTakingContext::UNKNOWN)) {
      ctfDir += fmt::format("{}_{}/", mDataTakingContext.envId, mDataTakingContext.runNumber);
      if (!ctfDir.empty()) {
        o2::utils::createDirectoriesIfAbsent(ctfDir);
        LOGP(info, "Created {} directory for CTFs output", ctfDir);
      }
    }
    mCurrentCTFFileName = o2::base::NameConf::getCTFFileName(mTimingInfo.runNumber, mTimingInfo.firstTForbit, mTimingInfo.tfCounter, mHostName);
    mCurrentCTFFileNameFull = fmt::format("{}{}", ctfDir, mCurrentCTFFileName);
    mCTFFileOut.reset(TFile::Open(fmt::format("{}{}", mCurrentCTFFileNameFull, TMPFileEnding).c_str(), "recreate")); // to prevent premature external usage, use temporary name
    mCTFTreeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");

    mNCTFFiles++;
  }
}

//___________________________________________________________________
void CTFWriterSpec::closeTFTreeAndFile()
{
  if (mCTFTreeOut) {
    try {
      mCTFFileOut->cd();
      mCTFTreeOut->Write();
      mCTFTreeOut.reset();
      mCTFFileOut->Close();
      mCTFFileOut.reset();
      if (!TMPFileEnding.empty()) {
        std::filesystem::rename(o2::utils::Str::concat_string(mCurrentCTFFileNameFull, TMPFileEnding), mCurrentCTFFileNameFull);
      }
      // write CTF file metaFile data
      if (mStoreMetaFile) {
        o2::dataformats::FileMetaData ctfMetaData;
        ctfMetaData.fillFileData(mCurrentCTFFileNameFull);
        ctfMetaData.setDataTakingContext(mDataTakingContext);
        ctfMetaData.type = mMetaDataType;
        ctfMetaData.priority = "high";
        ctfMetaData.tfOrbits.swap(mTFOrbits);
        auto metaFileNameTmp = fmt::format("{}{}.tmp", mCTFMetaFileDir, mCurrentCTFFileName);
        auto metaFileName = fmt::format("{}{}.done", mCTFMetaFileDir, mCurrentCTFFileName);
        try {
          std::ofstream metaFileOut(metaFileNameTmp);
          metaFileOut << ctfMetaData;
          metaFileOut.close();
          std::filesystem::rename(metaFileNameTmp, metaFileName);
        } catch (std::exception const& e) {
          LOG(error) << "Failed to store CTF meta data file " << metaFileName << ", reason: " << e.what();
        }
      }
    } catch (std::exception const& e) {
      LOG(error) << "Failed to finalize CTF file " << mCurrentCTFFileNameFull << ", reason: " << e.what();
    }
    mTFOrbits.clear();
    mNAccCTF = 0;
    mAccCTFSize = 0;
    removeLockFile();
  }
}

//___________________________________________________________________
void CTFWriterSpec::storeDictionaries()
{
  // monolitic dictionary in tree format
  mDictTimeStamp = uint32_t(std::time(nullptr));
  auto getFileName = [this](bool curr) {
    return fmt::format("{}{}Tree_{}_{}_{}.root", this->mDictDir, o2::base::NameConf::CTFDICT, DetID::getNames(this->mDets, '-'), curr ? this->mDictTimeStamp : this->mPrevDictTimeStamp, curr ? this->mNCTF : this->mNCTFPrevDict);
  };
  auto dictFileName = getFileName(true);
  mDictFileOut.reset(TFile::Open(dictFileName.c_str(), "recreate"));
  mDictTreeOut = std::make_unique<TTree>(std::string(o2::base::NameConf::CTFDICT).c_str(), "O2 CTF dictionary");

  CTFHeader header{mTimingInfo.runNumber, uint32_t(mNCTF)};
  storeDictionary<o2::itsmft::CTF>(DetID::ITS, header);
  storeDictionary<o2::itsmft::CTF>(DetID::MFT, header);
  storeDictionary<o2::tpc::CTF>(DetID::TPC, header);
  storeDictionary<o2::trd::CTF>(DetID::TRD, header);
  storeDictionary<o2::tof::CTF>(DetID::TOF, header);
  storeDictionary<o2::ft0::CTF>(DetID::FT0, header);
  storeDictionary<o2::fv0::CTF>(DetID::FV0, header);
  storeDictionary<o2::fdd::CTF>(DetID::FDD, header);
  storeDictionary<o2::mid::CTF>(DetID::MID, header);
  storeDictionary<o2::mch::CTF>(DetID::MCH, header);
  storeDictionary<o2::emcal::CTF>(DetID::EMC, header);
  storeDictionary<o2::phos::CTF>(DetID::PHS, header);
  storeDictionary<o2::cpv::CTF>(DetID::CPV, header);
  storeDictionary<o2::zdc::CTF>(DetID::ZDC, header);
  storeDictionary<o2::hmpid::CTF>(DetID::HMP, header);
  storeDictionary<o2::ctp::CTF>(DetID::CTP, header);
  mDictFileOut->cd();
  appendToTree(*mDictTreeOut.get(), "CTFHeader", header);
  mDictTreeOut->SetEntries(1);
  mDictTreeOut->Write(mDictTreeOut->GetName(), TObject::kSingleKey);
  mDictTreeOut.reset();
  mDictFileOut.reset();
  std::string dictFileNameLnk = fmt::format("{}{}.root", mDictDir, o2::base::NameConf::CTFDICT);
  if (std::filesystem::exists(dictFileNameLnk)) {
    std::filesystem::remove(dictFileNameLnk);
  }
  std::filesystem::create_symlink(dictFileName, dictFileNameLnk);
  LOGP(info, "Saved CTF dictionaries tree with {} TFs to {} and linked to {}", mNCTF, dictFileName, dictFileNameLnk);
  if (mPrevDictTimeStamp) {
    auto dictFileNamePrev = getFileName(false);
    if (std::filesystem::exists(dictFileNamePrev)) {
      std::filesystem::remove(dictFileNamePrev);
      LOGP(info, "Removed previous dictionary version {}", dictFileNamePrev);
    }
  }
  mNCTFPrevDict = mNCTF;
  mPrevDictTimeStamp = mDictTimeStamp;
}

//___________________________________________________________________
void CTFWriterSpec::createLockFile(int level)
{
  // create lock file for the CTF to be written to the storage of given level
  while (1) {
    mLockFileName = fmt::format("{}/ctfs{}-{}_{}_{}_{}.lock", LOCKFileDir, level, o2::utils::Str::getRandomString(8), mTimingInfo.runNumber, mTimingInfo.firstTForbit, mTimingInfo.tfCounter);
    if (!std::filesystem::exists(mLockFileName)) {
      break;
    }
  }
  mLockFD = open(mLockFileName.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
  if (mLockFD == -1) {
    throw std::runtime_error(fmt::format("Error opening lock file {}", mLockFileName));
  }
  if (lockf(mLockFD, F_LOCK, 0)) {
    throw std::runtime_error(fmt::format("Error locking file {}", mLockFileName));
  }
}

//___________________________________________________________________
void CTFWriterSpec::removeLockFile()
{
  // remove CTF lock file
  if (mLockFD != -1) {
    if (lockf(mLockFD, F_ULOCK, 0)) {
      throw std::runtime_error(fmt::format("Error unlocking file {}", mLockFileName));
    }
    mLockFD = -1;
    std::error_code ec;
    std::filesystem::remove(mLockFileName, ec); // use non-throwing version
  }
}

//___________________________________________________________________
size_t CTFWriterSpec::getAvailableDiskSpace(const std::string& path, int level)
{
  // count number of CTF files in processing (written to storage at given level) from their lock files
  std::regex pat{fmt::format("({}/ctfs{}-[[:alnum:]_]+\\.lock$)", LOCKFileDir, level)};
  int nLocked = 0;
  size_t written = 0;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(LOCKFileDir)) {
    const auto& entryName = entry.path().native();
    if (std::regex_search(entryName, pat) && (mLockFD < 0 || entryName != mLockFileName)) {
      int fdt = open(entryName.c_str(), O_RDONLY);
      if (fdt != -1) {
        bool locked = lockf(fdt, F_TEST, 0) != 0;
        if (locked) {
          nLocked++;
          size_t sz = 0;
          auto nrd = read(fdt, &sz, sizeof(size_t));
          if (nrd == sizeof(size_t)) {
            written += sz;
          }
        }
        close(fdt);
        // unlocked file is either leftover from crached job or a file from concurent job which was being locked
        // or just unlocked but not yet removed. In the former case remove it
        if (!locked) {
          struct stat statbuf;
          if (stat(entryName.c_str(), &statbuf) != -1) { // if we fail to stat, the file was already removed
#ifdef __APPLE__
            auto ftime = statbuf.st_mtimespec.tv_sec; // last write time
#else
            auto ftime = statbuf.st_mtim.tv_sec; // last write time
#endif
            auto ctime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (ftime + 60 < ctime) {                 // this is an old file, remove it
              std::filesystem::remove(entryName, ec); // use non-throwing version
            }
          }
        }
      }
    }
  }
  const auto si = std::filesystem::space(path, ec);
  int64_t avail = int64_t(si.available) - nLocked * mChkSize + written; // account already written part of unfinished files
  LOGP(debug, "{} CTF files open (curr.size: {}) -> can use {} of {} bytes", nLocked, written, avail, si.available);
  return avail > 0 ? avail : 0;
}

//___________________________________________________________________
DataProcessorSpec getCTFWriterSpec(DetID::mask_t dets, uint64_t run, const std::string& outType, int verbosity, int reportInterval)
{
  std::vector<InputSpec> inputs;
  LOG(debug) << "Detectors list:";
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (dets[id]) {
      inputs.emplace_back(DetID::getName(id), DetID::getDataOrigin(id), "CTFDATA", 0, Lifetime::Timeframe);
      LOG(debug) << "Det " << DetID::getName(id) << " added";
    }
  }
  return DataProcessorSpec{
    "ctf-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<CTFWriterSpec>(dets, run, outType, verbosity, reportInterval)}, // RS FIXME once global/local options clash is solved, --output-type will become device option
    Options{                                                                                    //{"output-type", VariantType::String, "ctf", {"output types: ctf (per TF) or dict (create dictionaries) or both or none"}},
            {"save-ctf-after", VariantType::Int, 0, {"if > 0, autosave CTF tree with multiple CTFs after every N CTFs"}},
            {"save-dict-after", VariantType::Int, 0, {"if > 0, in dictionary generation mode save it dictionary after certain number of TFs processed"}},
            {"ctf-dict-dir", VariantType::String, "none", {"CTF dictionary directory, must exist"}},
            {"output-dir", VariantType::String, "none", {"CTF output directory, must exist"}},
            {"output-dir-alt", VariantType::String, "/dev/null", {"Alternative CTF output directory, must exist (if not /dev/null)"}},
            {"meta-output-dir", VariantType::String, "/dev/null", {"CTF metadata output directory, must exist (if not /dev/null)"}},
            {"min-file-size", VariantType::Int64, 0l, {"accumulate CTFs until given file size reached"}},
            {"max-file-size", VariantType::Int64, 0l, {"if > 0, try to avoid exceeding given file size, also used for space check"}},
            {"max-ctf-per-file", VariantType::Int, 0, {"if > 0, avoid storing more than requested CTFs per file"}},
            {"ctf-rejection", VariantType::Int, 0, {">0: percentage to reject randomly, <0: reject if timeslice%|value|!=0"}},
            {"ignore-partition-run-dir", VariantType::Bool, false, {"Do not creare partition-run directory in output-dir"}}}};
}

} // namespace ctf
} // namespace o2
