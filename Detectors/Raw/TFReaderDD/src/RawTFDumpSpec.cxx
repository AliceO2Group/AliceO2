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

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Task.h"
#include "Framework/DataTakingContext.h"
#include "Framework/TimingInfo.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "RawTFDumpSpec.h"
#include "TFReaderDD/SubTimeFrameFile.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CommonUtils/StringUtils.h"
#include "Algorithm/RangeTokenizer.h"
#include <unistd.h>
#include <TMath.h>
#include <filesystem>
#include <random>
#include <set>

namespace o2::rawdd
{
namespace o2h = o2::header;
using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using DetID = o2::detectors::DetID;
using ios = std::ios_base;

class RawTFDump : public Task
{
 public:
  static constexpr o2h::DataDescription DESCRaw{"RAWDATA"}, DESCCRaw{"CRAWDATA"};

  RawTFDump(const std::string& trigger);
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  bool triggerTF(ProcessingContext& pc);
  void updateTimeDependentParams(ProcessingContext& pc);
  void prepareTFForWriting(ProcessingContext& pc);
  size_t getTFSizeInFile() const;
  size_t getCurrentFileSize();
  void prepareTFFile();
  void closeTFFile();
  bool checkFreeSpace(ProcessingContext& pc);
  std::string reportRates() const;

  SubTimeFrameFileDataIndex mTFDataIndex;
  std::vector<std::pair<const void*, const void*>> mTFData;
  std::map<EquipmentIdentifier, std::tuple<size_t, size_t, size_t>> mDataMap;
  std::vector<InputSpec> mFilter{};
  std::vector<InputSpec> mTriggerFilter{};
  std::vector<InputSpec> mExclTriggerFilter{};

  size_t mTFSize = 0;
  size_t mMinFileSize = 0; // if > 0, accumulate TFs in the same file until the total size exceeds this minimum
  size_t mMaxFileSize = 0; // if > MinSize, and accumulated size will exceed this value, stop accumulation (even if mMinFileSize is not reached)

  int mNTFsSeen = 0;          // total number of TFs seen
  int mNTFsExtTrig = 0;       // total nunber of TFs externally triggered
  int mNTFsAccepted = 0;      // total number of TFs written
  int mNTFsInFile = 0;        // total number of TFs accumulated in the current file
  int mNTFFiles = 0;          // total number of TF files written
  int mLastWarned = 0;        // TF when last warned about throttling
  int mMaxTFPerFile = 0;      // max TFs per files to store
  int mNWarnThrottle = 0;     // number of times we warned about the throttling
  int mMaxWarnThrottle = 0;   // max allowed warnings about the throttling
  int mWarnThrottleTF = 0;    // min period (in TFs) between the warnings about the throttling
  int mWaitDiskFull = 0;      // if mCheckDiskFull triggers, pause for this amount of ms before new attempt
  int mWaitDiskFullMax = -1;  // produce fatal mCheckDiskFull block the workflow for more than this time (in ms)
  float mCheckDiskFull = 0.;  // wait for if available abs. disk space is < mCheckDiskFull (if >0) or if its fraction is < -mCheckDiskFull (if <0)
  float mMaxAccRate = 0.f;    // max acceptance rate
  float mConfLim = 0.05f;     // confidence limit for rate esimate (lower quantile)
  float mRateEstAccLow = 0.f; // lower limit on accepted TFs rate
  float mRateEstAccUpp = 0.f; // upper limit on accepted TFs rate
  float mRateEstTrgLow = 0.f; // lower limit on triggered TFs rate
  float mRateEstTrgUpp = 0.f; // upper limit on triggered TFs rate

  bool mFillMD5 = false;
  bool mWriteTF = true; // for dry run
  bool mStoreMetaFile = false;
  bool mCreateRunEnvDir = true;
  bool mAcceptCurrentTF = false;
  bool mRejectDEADBEEF = false;
  bool mRejectDistSTF = true;
  int mVerbose = 0;
  std::vector<uint32_t> mTFOrbits{}; // 1st orbits of TF accumulated in current file
  o2::framework::DataTakingContext mDataTakingContext{};
  o2::framework::TimingInfo mTimingInfo{};

  std::string mTrigger{}; // external trigger input
  std::string mExclTriggerSpecs{}; // trigger specs to ignore
  std::string mHostName{};
  std::string mTFDir{};
  std::string mTFMetaFileDir = "/dev/null";
  std::string mCurrentTFFileName{};
  std::string mCurrentTFFileNameFull{};
  std::string mCurrentTFFileNameFullTmp{};
  std::string mMetaDataType{};

  static constexpr size_t MiB = 1ul << 20;
  static constexpr std::streamsize sBuffSize = MiB; // 1 MiB
  static constexpr std::streamsize sChunkSize = 512;
  static const std::string TMPFileEnding;
  std::unique_ptr<char[]> mFileBuf;
  std::ofstream mFile;
  std::uniform_real_distribution<double> mUniformDist{0.0, 100.0};
  std::default_random_engine mRGen;

  // helper to make sure the written blocks are buffered
  template <
    typename pointer,
    typename std::enable_if<
      std::is_pointer<pointer>::value &&                      // pointers only
      (std::is_void<std::remove_pointer_t<pointer>>::value || // void* or standard layout!
       std::is_standard_layout<std::remove_pointer_t<pointer>>::value)>::type* = nullptr>
  void buffered_write(const pointer p, std::streamsize pCount)
  {
    // make sure we're not doing a short write
    assert((pCount % sizeof(std::conditional_t<std::is_void<std::remove_pointer_t<pointer>>::value,
                                               char, std::remove_pointer_t<pointer>>) ==
            0) &&
           "Performing short write?");

    const char* lPtr = reinterpret_cast<const char*>(p);
    // avoid the optimization if the write is large enough
    if (pCount >= sBuffSize) {
      mFile.write(lPtr, pCount);
    } else {
      // split the write to smaller chunks
      while (pCount > 0) {
        const auto lToWrite = std::min(pCount, sChunkSize);
        assert(lToWrite > 0 && lToWrite <= sChunkSize && lToWrite <= pCount);

        mFile.write(lPtr, lToWrite);
        lPtr += lToWrite;
        pCount -= lToWrite;
      }
    }
  }
};

const std::string RawTFDump::TMPFileEnding{".part"};

//________________________________________
RawTFDump::RawTFDump(const std::string& trigger) : mTrigger{trigger}
{
  mTriggerFilter = select(trigger.c_str());
  mFileBuf = std::make_unique<char[]>(sBuffSize);
  mFile.rdbuf()->pubsetbuf(mFileBuf.get(), sBuffSize);
  mFile.clear();
  mFile.exceptions(std::fstream::failbit | std::fstream::badbit);
}

//________________________________________
void RawTFDump::init(InitContext& ic)
{
  mRGen = std::default_random_engine(getpid());
  mTFMetaFileDir = ic.options().get<std::string>("meta-output-dir");
  if (mTFMetaFileDir != "/dev/null") {
    mTFMetaFileDir = o2::utils::Str::rectifyDirectory(mTFMetaFileDir);
    mStoreMetaFile = true;
    mFillMD5 = ic.options().get<bool>("md5-for-meta");
  }

  mTFDir = ic.options().get<std::string>("output-dir");
  if (mTFDir != "/dev/null") {
    mTFDir = o2::utils::Str::rectifyDirectory(mTFDir);
    mWriteTF = true;
  } else {
    mWriteTF = false;
    mStoreMetaFile = false;
  }
  mRejectDistSTF = !ic.options().get<bool>("include-dist-stf");
  mRejectDEADBEEF = !ic.options().get<bool>("include-deadbeef");
  mCreateRunEnvDir = !ic.options().get<bool>("ignore-partition-run-dir");
  mMinFileSize = ic.options().get<int64_t>("min-file-size");
  mMaxFileSize = ic.options().get<int64_t>("max-file-size");
  mMaxTFPerFile = ic.options().get<int>("max-tf-per-file");
  mMaxAccRate = ic.options().get<float>("max-dump-rate");
  float cl = ic.options().get<float>("rate-est-conf-limit");
  if (mConfLim < 0.001 || mConfLim > 0.32) {
    LOGP(warn, "Bad confidence limit {} for rate estimate, setting to default {}", cl, mConfLim);
  } else {
    mConfLim = cl;
  }
  mMaxWarnThrottle = ic.options().get<int>("max-warn");
  mWarnThrottleTF = ic.options().get<int>("mute-warn-period");

  mVerbose = ic.options().get<int>("verbosity-level");
  mExclTriggerSpecs = ic.options().get<std::string>("exclude-trigger-specs");
  if (!mExclTriggerSpecs.empty()) {
    mExclTriggerFilter = select(mExclTriggerSpecs.c_str());
  }
  if (mTrigger.empty()) {
    if (mMaxAccRate >= 0.f) {
      LOGP(info, "Will accept randomly {}% of TFs", mMaxAccRate);
    } else {
      LOGP(info, "Will accept every {}-th TF", int(std::ceil(-100.f / mMaxAccRate)));
    }
  } else {
    mMaxAccRate = std::abs(mMaxAccRate);
    LOGP(info, "Will limit TFs triggered with {} by {}% at most", mTrigger, mMaxAccRate);
    if (!mExclTriggerFilter.empty()) {
      LOGP(info, "Inputs excluded from the trigger: {}", mExclTriggerSpecs);
    }
  }

  if (mWriteTF) {
    if (mMinFileSize > 0) {
      LOGP(info, "Multiple TFs will be accumulated in the file until its size exceeds {}{}",
           mMinFileSize, mMaxFileSize > mMinFileSize ? fmt::format(" but does not exceed {} B", mMaxFileSize) : std::string{});
    }
  }

  mCheckDiskFull = ic.options().get<float>("require-free-disk");
  mWaitDiskFull = 1000 * ic.options().get<float>("wait-for-free-disk");
  mWaitDiskFullMax = 1000 * ic.options().get<float>("max-wait-for-free-disk");

  char hostname[_POSIX_HOST_NAME_MAX];
  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  mHostName = hostname;
  mHostName = mHostName.substr(0, mHostName.find('.'));
}

//________________________________________
void RawTFDump::run(ProcessingContext& pc)
{
  mNTFsSeen++;
  updateTimeDependentParams(pc);
  mAcceptCurrentTF = triggerTF(pc);
  if (mAcceptCurrentTF) {
    prepareTFForWriting(pc);
  } else {
    return;
  }

  prepareTFFile();
  if (mWriteTF && checkFreeSpace(pc)) { // write data
    try {
      size_t lTFSizeInFile = getTFSizeInFile();
      SubTimeFrameFileMeta lTFFileMeta(lTFSizeInFile);
      lTFFileMeta.mWriteTimeMs = mTimingInfo.creation;

      mFile << lTFFileMeta;  // Write DataHeader + SubTimeFrameFileMeta
      mFile << mTFDataIndex; // Write DataHeader + SubTimeFrameFileDataIndex

      for (const auto& eqEntry : mDataMap) {
        auto& [lSize, lCnt, lEntry] = eqEntry.second;
        for (size_t part = 0; part < lCnt; part++) {
          const auto& dataPtr = mTFData[lEntry + part];
          DataHeader hdToWrite = *reinterpret_cast<const DataHeader*>(dataPtr.first); // make a local DataHeader copy to clear flagsNextHeader bit
          hdToWrite.flagsNextHeader = 0;
          hdToWrite.splitPayloadIndex = part;
          if (mVerbose > 2) {
            LOGP(info, "Writing part:{}/{} of {} | TFCounter:{} part{}/{}", part, lCnt, DataSpecUtils::describe(OutputSpec{hdToWrite.dataOrigin, hdToWrite.dataDescription, hdToWrite.subSpecification}), hdToWrite.firstTForbit, hdToWrite.splitPayloadIndex, hdToWrite.splitPayloadParts);
          }
          buffered_write(reinterpret_cast<const char*>(&hdToWrite), sizeof(DataHeader));
          buffered_write(dataPtr.second, hdToWrite.payloadSize);
        }
      }
      mFile.flush(); // flush the buffer and check the state
      mTFOrbits.push_back(mTimingInfo.firstTForbit);
      mNTFsInFile++;
    } catch (const std::ios_base::failure& eFailExc) {
      LOGP(error, "Writing of TF {} to file {} failed. error={}", mTimingInfo.tfCounter, mCurrentTFFileNameFullTmp, eFailExc.what());
    }
  }
  // cleanup
  mTFData.clear();
  mDataMap.clear();
  mTFDataIndex.clear();
  mTFSize = 0;
}

//____________________________________________________________
void RawTFDump::endOfStream(EndOfStreamContext&)
{
  closeTFFile();
  LOGP(info, "Dumped {} TFs to {} files", mNTFsAccepted, mNTFFiles);
  if (!mTriggerFilter.empty()) {
    LOGP(info, "External trigger summary: {}", reportRates());
  }
}

//________________________________________
size_t RawTFDump::getTFSizeInFile() const
{
  return SubTimeFrameFileMeta::getSizeInFile() + mTFDataIndex.getSizeInFile() + mTFSize;
}

//________________________________________
size_t RawTFDump::getCurrentFileSize()
{
  return mFile.is_open() ? size_t(mFile.tellp()) : 0;
}

//___________________________________________________________________
void RawTFDump::prepareTFFile()
{
  if (!mWriteTF) {
    return;
  }
  bool needToOpen;
  if (!mFile.is_open()) {
    needToOpen = true;
  } else {
    auto currSize = getCurrentFileSize();
    if ((mNTFsInFile >= mMaxTFPerFile) ||
        (currSize >= mMinFileSize) ||                                                         // min size exceeded, may close the file.
        (currSize && mMaxFileSize > mMinFileSize && ((currSize + mTFSize) > mMaxFileSize))) { // this is not the 1st TF in the file and the new size will exceed allowed max
      needToOpen = true;
    } else {
      LOGP(info, "Will add new TF of size {} to existing file of size {} with {} TFs", mTFSize, currSize, mNTFsInFile);
      needToOpen = false;
    }
  }
  if (needToOpen) {
    closeTFFile();
    auto TFDir = mTFDir.empty() ? o2::utils::Str::rectifyDirectory("./") : mTFDir;
    if (mCreateRunEnvDir && !mDataTakingContext.envId.empty() && (mDataTakingContext.envId != o2::framework::DataTakingContext::UNKNOWN)) {
      TFDir += fmt::format("{}_{}tf/", mDataTakingContext.envId, mDataTakingContext.runNumber);
      if (!TFDir.empty()) {
        o2::utils::createDirectoriesIfAbsent(TFDir);
        LOGP(info, "Created {} directory for TFs output", TFDir);
      }
    }
    mCurrentTFFileName = o2::base::NameConf::getRawTFFileName(mTimingInfo.runNumber, mTimingInfo.firstTForbit, mTimingInfo.tfCounter, mHostName);
    mCurrentTFFileNameFull = fmt::format("{}{}", TFDir, mCurrentTFFileName);
    mCurrentTFFileNameFullTmp = TMPFileEnding.empty() ? mCurrentTFFileNameFull : o2::utils::Str::concat_string(mCurrentTFFileNameFull, TMPFileEnding);
    mFile.open(mCurrentTFFileNameFullTmp.c_str(), ios::binary | ios::trunc | ios::out | ios::ate);
    LOGP(info, "Opened new raw-tf dump file {}[{}]", mCurrentTFFileNameFull, TMPFileEnding);
    mNTFFiles++;
  }
}

//___________________________________________________________________
void RawTFDump::updateTimeDependentParams(ProcessingContext& pc)
{
  namespace GRPECS = o2::parameters::GRPECS;
  mTimingInfo = pc.services().get<o2::framework::TimingInfo>();
  if (mTimingInfo.globalRunNumberChanged) {
    mDataTakingContext = pc.services().get<DataTakingContext>();
    // determine the output type for the TF metadata
    mMetaDataType = GRPECS::getRawDataPersistencyMode(mDataTakingContext.runType, mDataTakingContext.forcedRaw);
  }
}

//___________________________________________________________________
void RawTFDump::closeTFFile()
{
  if (!mFile.is_open()) {
    return;
  }
  try {
    LOGP(info, "Closing output file {}[{}]", mCurrentTFFileNameFull, TMPFileEnding);
    mFile.close();
    // write TF file metaFile data
    if (mStoreMetaFile) {
      o2::dataformats::FileMetaData TFMetaData;
      if (!TFMetaData.fillFileData(mCurrentTFFileNameFullTmp, mFillMD5, TMPFileEnding)) {
        throw std::runtime_error("metadata file was requested but not created");
      }
      TFMetaData.setDataTakingContext(mDataTakingContext);
      TFMetaData.type = mMetaDataType;
      TFMetaData.priority = "high";
      TFMetaData.tfOrbits.swap(mTFOrbits);
      auto metaFileNameTmp = fmt::format("{}{}.tmp", mTFMetaFileDir, mCurrentTFFileName);
      auto metaFileName = fmt::format("{}{}.done", mTFMetaFileDir, mCurrentTFFileName);
      try {
        std::ofstream metaFileOut(metaFileNameTmp);
        metaFileOut << TFMetaData;
        metaFileOut.close();
        if (!TMPFileEnding.empty()) {
          std::filesystem::rename(mCurrentTFFileNameFullTmp, mCurrentTFFileNameFull);
        }
        std::filesystem::rename(metaFileNameTmp, metaFileName);
        LOGP(info, "wrote meta file {}", metaFileName);
      } catch (std::exception const& e) {
        LOGP(error, "Failed to store TF meta data file {}, reason {}", metaFileName, e.what());
      }
    } else if (!TMPFileEnding.empty()) {
      std::filesystem::rename(mCurrentTFFileNameFullTmp, mCurrentTFFileNameFull);
    }
  } catch (std::exception const& e) {
    LOGP(error, "Failed to finalize TF file {}, reason: ", mCurrentTFFileNameFull, e.what());
  }
  mTFOrbits.clear();
  mNTFsInFile = 0;
}

//________________________________________
bool RawTFDump::checkFreeSpace(ProcessingContext& pc)
{
  int totalWait = 0, nwaitCycles = 0;
  while (mCheckDiskFull) {
    constexpr int showFirstN = 10, prsecaleWarnings = 50;
    try {
      const auto si = std::filesystem::space(mCurrentTFFileNameFullTmp);
      std::string wmsg{};
      if (mCheckDiskFull > 0.f && si.available < mCheckDiskFull) {
        nwaitCycles++;
        wmsg = fmt::format("Disk has {} MiB available while at least {} MiB is requested, wait for {} ms (on top of {} ms)", si.available / MiB, size_t(mCheckDiskFull) / MiB, mWaitDiskFull, totalWait);
      } else if (mCheckDiskFull < 0.f && float(si.available) / si.capacity < -mCheckDiskFull) { // relative margin requested
        nwaitCycles++;
        wmsg = fmt::format("Disk has {:.3f}% available while at least {:.3f}% is requested, wait for {} ms (on top of {} ms)", si.capacity ? float(si.available) / si.capacity * 100.f : 0., -mCheckDiskFull, mWaitDiskFull, totalWait);
      } else {
        nwaitCycles = 0;
      }
      if (nwaitCycles) {
        if (mWaitDiskFullMax > 0 && totalWait > mWaitDiskFullMax) {
          closeTFFile(); // try to save whatever we have
          LOGP(fatal, "Disk has {} MiB available out of {} MiB after waiting for {} ms", si.available / MiB, si.capacity / MiB, mWaitDiskFullMax);
        }
        if (nwaitCycles < showFirstN + 1 || (prsecaleWarnings && (nwaitCycles % prsecaleWarnings) == 0)) {
          LOGP(alarm, "{}", wmsg);
        }
        pc.services().get<RawDeviceService>().waitFor((unsigned int)(mWaitDiskFull));
        totalWait += mWaitDiskFull;
        continue;
      }
    } catch (std::exception const& e) {
      LOGP(fatal, "unable to query disk space info for path {}, reason {}", mCurrentTFFileNameFull, e.what()); // do we want this?
    }
    break;
  }
  return true;
}

//________________________________________
bool RawTFDump::triggerTF(ProcessingContext& pc)
{
  bool trig = false;
  if (mTrigger.empty()) { // random
    if (mMaxAccRate > 0.f) {
      trig = (mUniformDist(mRGen) <= mMaxAccRate);
    } else if (mMaxAccRate < 0.f) {
      trig = (mTimingInfo.tfCounter % int(std::ceil(-100.f / mMaxAccRate))) == 0;
    }
  } else {
    for (auto const& ref : InputRecordWalker(pc.inputs(), mTriggerFilter)) {
      auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
      if (!dh) {
        LOGP(error, "Failed to extract header for trigger input");
        continue;
      }
      auto extTrig = DataRefUtils::as<bool>(ref);
      if (mVerbose > 0) {
        LOGP(info, "trigger input {}, part: {} of {}, payload {}, 1stTFOrbit: {} TF: {} | span size: {} span[0]={}",
             DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}),
             dh->splitPayloadIndex, dh->splitPayloadParts, dh->payloadSize, dh->firstTForbit, dh->tfCounter, extTrig.size(), extTrig.size() > 0 ? extTrig[0] : false);
      }
      if (extTrig.size() && extTrig[0]) {
        // is the input with this trigger vetoed?
        bool veto = false;
        for (const auto& excl : mExclTriggerFilter) {
          if (DataRefUtils::match(ref, excl)) {
            if (mVerbose > 0) {
              LOGP(info, "ignoring trigger from black-listed {}", DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}));
            }
            veto = true;
            break;
          }
        }
        if (veto) {
          continue;
        }
        trig = true;
        break;
      }
    }
    if (trig) { // do we need to throttle?
      mNTFsExtTrig++;
      mRateEstTrgLow = TMath::ChisquareQuantile(mConfLim, 2 * (mNTFsExtTrig)) / (2 * mNTFsSeen);
      mRateEstTrgUpp = TMath::ChisquareQuantile(1. - mConfLim, 2 * (mNTFsExtTrig + 1)) / (2 * mNTFsSeen);
      mRateEstAccLow = TMath::ChisquareQuantile(mConfLim, 2 * (mNTFsAccepted)) / (2 * mNTFsSeen);
      mRateEstAccUpp = TMath::ChisquareQuantile(1. - mConfLim, 2 * (mNTFsAccepted + 1)) / (2 * mNTFsSeen);
      if (mRateEstAccLow > 0.01 * mMaxAccRate) { // current lowest estimate on the acceptance rate exceeds desired limit -> ignore trigger
        trig = false;
        // do we need to warn?
        if ((mNTFsSeen - mLastWarned) > mWarnThrottleTF && ((mNWarnThrottle < mMaxWarnThrottle) || mMaxWarnThrottle < 0)) {
          mLastWarned = mNTFsSeen;
          std::string swarn = reportRates();
          if (++mNWarnThrottle == mMaxWarnThrottle) {
            swarn += " Will not warn anymore.";
          } else {
            swarn += fmt::format(" Will suppress this warnings for {} TFs", mWarnThrottleTF);
          }
          LOGP(alarm, "Ignoring TF triggered for dumping: {}", swarn);
        }
      }
    }
  }
  if (trig) {
    mNTFsAccepted++;
  }
  if (mVerbose > 0) {
    LOGP(info, "TF#{} (slice#{}) will{} be written, {}", mTimingInfo.tfCounter, mTimingInfo.timeslice, trig ? "" : " not", reportRates());
  }
  return trig;
}

//________________________________________
void RawTFDump::prepareTFForWriting(ProcessingContext& pc)
{
  for (auto const& ref : InputRecordWalker(pc.inputs(), mFilter)) {
    auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
    if (!dh) {
      LOGP(error, "Failed to extract header");
      continue;
    }
    if ((dh->subSpecification == 0xdeadbeef && mRejectDEADBEEF) ||
        (dh->dataOrigin == o2::header::gDataOriginFLP && dh->dataDescription == o2::header::gDataDescriptionDISTSTF && mRejectDistSTF)) {
      if (mVerbose > 2) {
        LOGP(info, "Rejecting {}", DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}));
      }
      continue;
    }
    const auto lHdrDataSize = sizeof(DataHeader) + dh->payloadSize;
    mTFSize += lHdrDataSize;

    auto& [lSize, lCnt, lEntry] = mDataMap[EquipmentIdentifier(*dh)];
    if (!lCnt) {
      lEntry = mTFData.size(); // flag where the data of this spec starts
    }
    lSize += lHdrDataSize;
    lCnt++;
    mTFData.push_back({ref.header, ref.payload});
    if (mVerbose > 2) {
      const auto* dph = DataRefUtils::getHeader<DataProcessingHeader*>(ref);
      LOGP(info, "{}, part: {} of {}, payload {}, 1stTFOrbit: {} TF: {}, creation: {} | counter:{} size:{} entry:{}",
           DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}),
           dh->splitPayloadIndex, dh->splitPayloadParts, dh->payloadSize, dh->firstTForbit, dh->tfCounter, dph ? dph->creation : -1UL, lCnt, lSize, lEntry);
    }
  }

  // build the index
  {
    LOGP(info, "Creating dump image for TF {} of run {}, starting orbit {}, size = {}", mTimingInfo.tfCounter, mTimingInfo.runNumber, mTimingInfo.firstTForbit, mTFSize);
    std::uint64_t lCurrOff = 0;
    for (const auto& eqEntry : mDataMap) {
      const auto& eq = eqEntry.first;
      auto& [lSize, lCnt, lEntry] = eqEntry.second;
      assert(lSize > sizeof(DataHeader));

      OutputSpec spec{eq.mDataOrigin, eq.mDataDescription, eq.mSubSpecification};
      if (mVerbose > 1) {
        LOGP(info, "{} : {} parts of size {} entry {}| offset: {}", DataSpecUtils::describe(spec), lCnt, lSize, lEntry, lCurrOff);
      }
      mTFDataIndex.AddStfElement(eq, lCnt, lCurrOff, lSize);
      lCurrOff += lSize;
    }
  }
}

//____________________________________________________________
std::string RawTFDump::reportRates() const
{
  std::string rep = fmt::format("{} TFs seen, {} accepted", mNTFsSeen, mNTFsAccepted);
  if (!mTrigger.empty()) {
    rep += fmt::format(", {} ext.triggered,  est.rate: [{:.2e}:{:.2e}]/[{:.2e}:{:.2e}].", mNTFsExtTrig, mRateEstAccLow, mRateEstAccUpp, mRateEstTrgLow, mRateEstTrgUpp);
  }
  return rep;
}

//__________________________________________________________
DataProcessorSpec getRawTFDumpSpec(const std::string& inpconfig, const std::string& trigger)
{
  std::vector<InputSpec> inputs = select(inpconfig.c_str());
  return DataProcessorSpec{
    "raw-tf-dump",
    inputs,
    {},
    AlgorithmSpec{adaptFromTask<RawTFDump>(trigger)},
    Options{
      {"include-deadbeef", VariantType::Bool, false, {"Include DPL-generated 0xdeadbeef subspecs for missing data"}},
      {"include-dist-stf", VariantType::Bool, false, {"Include FLP/DISTSUBTIMEFRAME input"}},
      {"exclude-trigger-specs", VariantType::String, "", {"Ignore trigger seen in these inputs of triggerspec"}},
      {"max-dump-rate", VariantType::Float, 0.f, {"%-age of TFs to dump. W/o external trigger: random(>0) or periodic(<0) rejection, with: max limit"}},
      {"rate-est-conf-limit", VariantType::Float, 0.05f, {"quantile for the lowest rate estimate confidence limit"}},
      {"max-warn", VariantType::Int, 5, {"max allowed warnings on throttling"}},
      {"mute-warn-period", VariantType::Int, 100, {"mute warnings on throttling for this number of TFs"}},
      {"output-dir", VariantType::String, "none", {"TF output directory, must exist"}},
      {"meta-output-dir", VariantType::String, "/dev/null", {"TF metadata output directory, must exist (if not /dev/null)"}},
      {"md5-for-meta", VariantType::Bool, false, {"fill CTF file MD5 sum in the metadata file"}},
      {"min-file-size", VariantType::Int64, 0l, {"accumulate TFs until given file size reached"}},
      {"max-file-size", VariantType::Int64, 0l, {"if > 0, try to avoid exceeding given file size, also used for space check"}},
      {"max-tf-per-file", VariantType::Int, 0, {"if > 0, avoid storing more than requested CTFs per file"}},
      {"require-free-disk", VariantType::Float, 0.f, {"pause writing op. if available disk space is below this margin, in bytes if >0, as a fraction of total if <0"}},
      {"wait-for-free-disk", VariantType::Float, 10.f, {"if paused due to the low disk space, recheck after this time (in s)"}},
      {"max-wait-for-free-disk", VariantType::Float, 60.f, {"produce fatal if paused due to the low disk space for more than this amount in s."}},
      {"verbosity-level", VariantType::Int, 0, {"Verbose mode: 1: decision on every TF, 2: details of saved TF, 3: more details"}},
      {"ignore-partition-run-dir", VariantType::Bool, false, {"Do not creare partition-run directory in output-dir"}}}};
}

} // namespace o2::rawdd
