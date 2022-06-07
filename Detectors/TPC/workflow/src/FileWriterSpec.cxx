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

/// \file FileWriterSpec.cxx
/// \brief  Writer for calibration data
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
//
#include <bitset>
#include <filesystem>
#include <memory>
#include <vector>
#include <string>
#include "fmt/format.h"

#include "TTree.h"

#include <fairmq/Device.h>
#include "Framework/Task.h"
#include "Framework/RawDeviceService.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataTakingContext.h"

#include "Headers/DataHeader.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "DetectorsCommonDataFormats/DetID.h"

#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCWorkflow/FileWriterSpec.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/KrCluster.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "TPCBase/Sector.h"

using namespace o2::framework;
using o2::dataformats::FileMetaData;
using SubSpecificationType = DataAllocator::SubSpecificationType;
using DetID = o2::detectors::DetID;

namespace o2::tpc
{

template <typename T>
class FileWriterDevice : public Task
{
 public:
  FileWriterDevice(const BranchType branchType, unsigned long sectorMask)
  {
    mBranchType = branchType;
    mSectorMask = sectorMask;
  }

  void init(InitContext& ic) final
  {
    mOutDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("output-dir"));

    mCreateRunEnvDir = !ic.options().get<bool>("ignore-partition-run-dir");
    mMaxTFPerFile = ic.options().get<int>("max-tf-per-file");
    mMetaFileDir = ic.options().get<std::string>("meta-output-dir");
    if (mMetaFileDir != "/dev/null") {
      mMetaFileDir = o2::utils::Str::rectifyDirectory(mMetaFileDir);
      mStoreMetaFile = true;
    }
  }

  void fillData(bool finalFill = false)
  {
    for (auto it = mCollectedData.begin(); it != mCollectedData.end();) {
      auto& dataStore = it->second;
      if (!finalFill && (dataStore.received.to_ulong() != mSectorMask)) {
        ++it;
        continue;
      }
      const auto oldTF = mPresentTF;
      mPresentTF = it->first;
      const auto oldTForbit = mFirstTForbit;
      mFirstTForbit = dataStore.firstOrbit;
      mTFOrbits.emplace_back(mFirstTForbit);

      LOGP(info, "Filling tree entry {} for run {}, tf {}, orbit {}, final {}, sectors {}, (expected: {})", mNTFs, mRun, mPresentTF, mFirstTForbit, finalFill, dataStore.received.to_string(), std::bitset<Sector::MAXSECTOR>(mSectorMask).to_string());
      auto& data = dataStore.data;
      std::array<std::vector<T>*, Sector::MAXSECTOR> dataPtr;
      for (size_t sector = 0; sector < data.size(); ++sector) {
        auto& inData = data[sector];
        dataPtr[sector] = &inData;

        if (!mDataBranches[sector]) {
          mDataBranches[sector] = mTreeOut->Branch(fmt::format("{}_{}", BranchName.at(mBranchType), sector).data(), &inData);
        } else {
          mDataBranches[sector]->SetAddress(&dataPtr[sector]);
        }
      }

      mTreeOut->Fill();
      for (size_t sector = 0; sector < data.size(); ++sector) {
        mDataBranches[sector]->ResetAddress();
      }

      mPresentTF = oldTF;
      mFirstTForbit = oldTForbit;
      ++mNTFs;

      it = mCollectedData.erase(it);
    }
  }

  void run(ProcessingContext& pc) final
  {
    static bool initOnceDone = false;
    if (!initOnceDone) {
      initOnceDone = true;
      mDataTakingContext = pc.services().get<DataTakingContext>();
    }

    const std::string NAStr = "NA";

    const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true));
    auto oldRun = mRun;
    auto oldTF = mPresentTF;
    mRun = processing_helpers::getRunNumber(pc);
    mPresentTF = dh->tfCounter;
    mFirstTForbit = dh->firstTForbit;

    if (mWrite) {
      if (!mCollectedData.size() || mCollectedData.find(mPresentTF) == mCollectedData.end()) {
        prepareTreeAndFile(dh);
      }

      fillData();
    }

    for (auto const& inputRef : InputRecordWalker(pc.inputs())) {
      auto& dataStore = mCollectedData[mPresentTF];
      auto const* sectorHeader = DataRefUtils::getHeader<TPCSectorHeader*>(inputRef);
      if (sectorHeader == nullptr) {
        LOGP(error, "sector header missing on header stack for input on ", inputRef.spec->binding);
        continue;
      }

      const int sector = sectorHeader->sector();
      // check if data was already received. Should never happen!
      if (dataStore.received.test(sector)) {
        LOGP(fatal, "data for sector {} and TF {} already received", sector, mPresentTF);
      }
      auto data = pc.inputs().get<std::vector<T>>(inputRef);
      // LOGP(info, "Received data for sector {} with {} entries in TF {}, orbit {}", sector, data.size(), mPresentTF, mFirstTForbit);
      dataStore.data[sector] = data;
      dataStore.received.set(sector);
      dataStore.firstOrbit = mFirstTForbit;
    }
  }

  void endOfStream(EndOfStreamContext& ec) final
  {
    fillData(true);
    closeTreeAndFile();
  }

  void stop() final
  {
    fillData(true);
    closeTreeAndFile();
  }

 private:
  struct DataStore {
    uint32_t firstOrbit{};
    std::array<std::vector<T>, Sector::MAXSECTOR> data;
    std::bitset<Sector::MAXSECTOR> received;
  };
  std::map<uint32_t, DataStore> mCollectedData;            ///< collected data per TF
  std::array<TBranch*, Sector::MAXSECTOR> mDataBranches{}; ///< data branch pointers
  std::vector<TBranch*> mInfoBranches;                     ///< common information
  std::vector<uint32_t> mTFOrbits{};                       ///< 1st orbits of TF accumulated in current file
  std::unique_ptr<TFile> mFileOut;                         ///< output file containin the tree
  std::unique_ptr<TTree> mTreeOut;                         ///< output tree
  std::unique_ptr<FileMetaData> mFileMetaData;             ///< meta data file for eos and alien file creation
  std::string mOutDir{};                                   ///< file output direcotry
  std::string mMetaFileDir{"/dev/null"};                   ///< output directory for meta data file
  std::string mCurrentFileName{};                          ///< current file name
  std::string mCurrentFileNameFull{};                      ///< current file name with full directory
  uint64_t mRun = 0;                                       ///< present run number
  uint32_t mPresentTF = 0;                                 ///< present TF number
  uint32_t mFirstTForbit = 0;                              ///< first orbit of present tf
  unsigned long mSectorMask = 0xFFFFFFFFF;                 ///< mask of configured sectors
  size_t mNTFs = 0;                                        ///< total number of TFs accumulated in the current file
  size_t mNFiles = 0;                                      ///< total number of calibration files written
  int mMaxTFPerFile = 0;                                   ///< maximum number of TFs per file
  bool mStoreMetaFile = false;                             ///< store the meata data file?
  bool mWrite = true;                                      ///< write data
  bool mCreateRunEnvDir = true;                            ///< create the output directory structure?
  BranchType mBranchType;                                  ///< output branch type
  o2::framework::DataTakingContext mDataTakingContext{};
  static constexpr std::string_view TMPFileEnding{".part"};

  void prepareTreeAndFile(const o2::header::DataHeader* dh);
  void closeTreeAndFile();
};
//___________________________________________________________________
template <typename T>
void FileWriterDevice<T>::prepareTreeAndFile(const o2::header::DataHeader* dh)
{
  if (!mWrite) {
    return;
  }
  bool needToOpen = false;
  if (!mTreeOut) {
    needToOpen = true;
  } else {
    if ((mMaxTFPerFile > 0) && (mNTFs >= mMaxTFPerFile)) {
      needToOpen = true;
    }
  }
  if (needToOpen) {
    LOGP(info, "opening new file");
    closeTreeAndFile();
    // auto fname = o2::base::NameConf::getCTFFileName(mRun, dh->firstTForbit, dh->tfCounter, "tpc_krypton");
    auto ctfDir = mOutDir.empty() ? o2::utils::Str::rectifyDirectory("./") : mOutDir;
    // if (mChkSize > 0 && (mDirFallBack != "/dev/null")) {
    // createLockFile(dh, 0);
    // auto sz = getAvailableDiskSpace(ctfDir, 0); // check main storage
    // if (sz < mChkSize) {
    // removeLockFile();
    // LOG(warning) << "Primary  output device has available size " << sz << " while " << mChkSize << " is requested: will write on secondary one";
    // ctfDir = mDirFallBack;
    // }
    // }
    if (mCreateRunEnvDir && !mDataTakingContext.envId.empty() && (mDataTakingContext.envId != o2::framework::DataTakingContext::UNKNOWN)) {
      ctfDir += fmt::format("{}_{}/", mDataTakingContext.envId, mDataTakingContext.runNumber);
      if (!std::filesystem::exists(ctfDir)) {
        if (!std::filesystem::create_directories(ctfDir)) {
          throw std::runtime_error(fmt::format("Failed to create {} directory", ctfDir));
        } else {
          LOG(info) << "Created {} directory for s output" << ctfDir;
        }
      }
    }
    mCurrentFileName = o2::base::NameConf::getCTFFileName(mRun, dh->firstTForbit, dh->tfCounter, "tpc_krypton");
    mCurrentFileNameFull = fmt::format("{}{}", ctfDir, mCurrentFileName);
    mFileOut.reset(TFile::Open(fmt::format("{}{}", mCurrentFileNameFull, TMPFileEnding).c_str(), "recreate")); // to prevent premature external usage, use temporary name
    mTreeOut = std::make_unique<TTree>(TreeName.at(mBranchType).data(), "O2 tree");
    mInfoBranches.emplace_back(mTreeOut->Branch("run", &mRun));
    mInfoBranches.emplace_back(mTreeOut->Branch("tfCounter", &mPresentTF));
    mInfoBranches.emplace_back(mTreeOut->Branch("firstOrbit", &mFirstTForbit));
    LOGP(info, "created {} info branches", mInfoBranches.size());
    if (mStoreMetaFile) {
      mFileMetaData = std::make_unique<o2::dataformats::FileMetaData>();
    }

    mNFiles++;
  }
}
//___________________________________________________________________
template <typename T>
void FileWriterDevice<T>::closeTreeAndFile()
{
  if (!mTreeOut) {
    return;
  }

  LOGP(info, "closing file {}", mCurrentFileName);
  try {
    mFileOut->cd();
    mTreeOut->SetEntries();
    mTreeOut->Write();
    mTreeOut.reset();
    mFileOut->Close();
    mFileOut.reset();
    if (!TMPFileEnding.empty()) {
      std::filesystem::rename(o2::utils::Str::concat_string(mCurrentFileNameFull, TMPFileEnding), mCurrentFileNameFull);
    }
    // write  file metaFile data
    if (mStoreMetaFile) {
      mFileMetaData->fillFileData(mCurrentFileNameFull);
      mFileMetaData->setDataTakingContext(mDataTakingContext);
      mFileMetaData->type = "raw";
      auto metaFileNameTmp = fmt::format("{}{}.tmp", mMetaFileDir, mCurrentFileName);
      auto metaFileName = fmt::format("{}{}.done", mMetaFileDir, mCurrentFileName);
      try {
        std::ofstream metaFileOut(metaFileNameTmp);
        metaFileOut << *mFileMetaData.get();
        metaFileOut << "TFOrbits: ";
        for (size_t i = 0; i < mTFOrbits.size(); i++) {
          metaFileOut << fmt::format("{}{}", i ? ", " : "", mTFOrbits[i]);
        }
        metaFileOut << '\n';
        metaFileOut.close();
        std::filesystem::rename(metaFileNameTmp, metaFileName);
      } catch (std::exception const& e) {
        LOG(error) << "Failed to store  meta data file " << metaFileName << ", reason: " << e.what();
      }
      mFileMetaData.reset();
    }
  } catch (std::exception const& e) {
    LOG(error) << "Failed to finalize  file " << mCurrentFileNameFull << ", reason: " << e.what();
  }
  mTFOrbits.clear();
  mInfoBranches.clear();
  std::fill(mDataBranches.begin(), mDataBranches.end(), nullptr);
  mNTFs = 0;
  // mAccSize = 0;
  // removeLockFile();
}

template <typename T>
DataProcessorSpec getFileWriterSpec(const std::string inputSpec, const BranchType branchType, unsigned long sectorMask)
{
  return DataProcessorSpec{
    "file-writer",
    select(inputSpec.data()),
    Outputs{},
    AlgorithmSpec{adaptFromTask<FileWriterDevice<T>>(branchType, sectorMask)},
    Options{
      {"output-dir", VariantType::String, "none", {" output directory, must exist"}},
      {"meta-output-dir", VariantType::String, "/dev/null", {" metadata output directory, must exist (if not /dev/null)"}},
      {"max-tf-per-file", VariantType::Int, 0, {"if > 0, avoid storing more than requested TFs per file"}},
      {"ignore-partition-run-dir", VariantType::Bool, false, {"Do not creare partition-run directory in output-dir"}},
    }};
}; // end DataProcessorSpec

// template spacializations
template o2::framework::DataProcessorSpec getFileWriterSpec<o2::tpc::Digit>(const std::string inputSpec, const BranchType branchType, unsigned long sectorMask);
template o2::framework::DataProcessorSpec getFileWriterSpec<o2::tpc::KrCluster>(const std::string inputSpec, const BranchType branchType, unsigned long sectorMask);
} // namespace o2::tpc
