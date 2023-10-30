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

#ifndef O2_TRD_KRWRITERSPEC_H
#define O2_TRD_KRWRITERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataTakingContext.h"
#include "DataFormatsTRD/KrCluster.h"
#include "DataFormatsTRD/KrClusterTriggerRecord.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"

#include <TFile.h>
#include <TTree.h>
#include <filesystem>
#include <mutex>
#include <fmt/format.h>

namespace o2
{
namespace trd
{

class TRDKrClsWriterTask : public o2::framework::Task
{
 public:
  TRDKrClsWriterTask() = default;

  void init(o2::framework::InitContext& ic) final
  {
    mOutputDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("output-dir"));

    // should we write meta files for epn2eos?
    mMetaFileDir = ic.options().get<std::string>("meta-output-dir");
    if (mMetaFileDir != "/dev/null") {
      mMetaFileDir = o2::utils::Str::rectifyDirectory(mMetaFileDir);
      mStoreMetaFile = true;
    }

    LOGP(info, "Storing output in {}, meta file writing enabled: {}", mOutputDir, mStoreMetaFile);
    mAutoSave = ic.options().get<int>("autosave-interval");

    char hostname[_POSIX_HOST_NAME_MAX];
    gethostname(hostname, _POSIX_HOST_NAME_MAX);
    mHostName = hostname;
    mHostName = mHostName.substr(0, mHostName.find('.'));
  }

  void createOutputFile(int runNumber, o2::framework::ProcessingContext& pc)
  {
    mFileName = fmt::format("o2_trdKrCls_run{}_{}.root", runNumber, mHostName);
    auto fileNameTmp = o2::utils::Str::concat_string(mOutputDir, mFileName, ".part");
    mFileOut = std::make_unique<TFile>(fileNameTmp.c_str(), "recreate");
    mTreeOut = std::make_unique<TTree>("krData", "TRD krypton cluster data");
    mTreeOut->Branch("cluster", &krClusterPtr);
    mTreeOut->Branch("trigRec", &krTrigRecPtr);
    mDataTakingContext = pc.services().get<DataTakingContext>();
    mOutputFileCreated = true;
  }

  void writeToFile()
  {
    if (!mOutputFileCreated) {
      return;
    }
    mFileOut->cd();
    mTreeOut->Write();
  }

  void closeOutputFile()
  {
    if (!mOutputFileCreated || mOutputFileClosed) {
      return;
    }
    std::lock_guard<std::mutex> guard(mMutex);
    writeToFile();
    mTreeOut.reset();
    mFileOut->Close();
    mFileOut.reset();
    auto fileNameWithPath = mOutputDir + mFileName;
    std::filesystem::rename(o2::utils::Str::concat_string(mOutputDir, mFileName, ".part"), fileNameWithPath);
    if (mStoreMetaFile) {
      o2::dataformats::FileMetaData fileMetaData; // object with information for meta data file
      fileMetaData.fillFileData(fileNameWithPath);
      fileMetaData.setDataTakingContext(mDataTakingContext);
      fileMetaData.type = "calib";
      fileMetaData.priority = "high";
      auto metaFileNameTmp = fmt::format("{}{}.tmp", mMetaFileDir, mFileName);
      auto metaFileName = fmt::format("{}{}.done", mMetaFileDir, mFileName);
      try {
        std::ofstream metaFileOut(metaFileNameTmp);
        metaFileOut << fileMetaData;
        metaFileOut.close();
        std::filesystem::rename(metaFileNameTmp, metaFileName);
      } catch (std::exception const& e) {
        LOG(error) << "Failed to store meta data file " << metaFileName << ", reason: " << e.what();
      }
    }
    mOutputFileClosed = true;
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    if (!mOutputFileCreated) {
      auto tInfo = pc.services().get<o2::framework::TimingInfo>();
      createOutputFile(tInfo.runNumber, pc);
    }
    if (mRunStopRequested) {
      return;
    }
    if (pc.transitionState() == TransitionHandlingState::Requested) {
      LOG(info) << "Run stop requested, closing output file";
      mRunStopRequested = true;
      closeOutputFile();
      return;
    }
    auto cluster = pc.inputs().get<gsl::span<KrCluster>>("krcluster");
    auto triggerRecords = pc.inputs().get<gsl::span<KrClusterTriggerRecord>>("krtrigrec");
    for (const auto& cls : cluster) {
      krCluster.push_back(cls);
    }
    for (const auto& trig : triggerRecords) {
      krTrigRec.push_back(trig);
    }
    mTreeOut->Fill();
    krCluster.clear();
    krTrigRec.clear();
    if (mAutoSave > 0 && ++mTFCounter % mAutoSave == 0) {
      writeToFile();
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    if (mRunStopRequested) {
      return;
    }
    LOG(info) << "End of stream received, closing output file";
    closeOutputFile();
  }

 private:
  bool mRunStopRequested{false};
  bool mStoreMetaFile{false};
  bool mOutputFileCreated{false};
  bool mOutputFileClosed{false};
  int mAutoSave{0};
  uint64_t mTFCounter{0};
  std::mutex mMutex;
  std::string mOutputDir{"none"};
  std::string mMetaFileDir{"/dev/null"};
  std::string mHostName{};
  std::string mFileName{};
  std::unique_ptr<TFile> mFileOut{};
  std::unique_ptr<TTree> mTreeOut{};
  std::vector<KrCluster> krCluster, *krClusterPtr{&krCluster};
  std::vector<KrClusterTriggerRecord> krTrigRec, *krTrigRecPtr{&krTrigRec};
  o2::framework::DataTakingContext mDataTakingContext;
};

framework::DataProcessorSpec getKrClusterWriterSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("krcluster", "TRD", "KRCLUSTER");
  inputs.emplace_back("krtrigrec", "TRD", "TRGKRCLS");

  return DataProcessorSpec{"kr-cluster-writer",
                           inputs,
                           Outputs{},
                           AlgorithmSpec{adaptFromTask<o2::trd::TRDKrClsWriterTask>()},
                           Options{
                             {"output-dir", VariantType::String, "none", {"Output directory for data. Defaults to current working directory"}},
                             {"meta-output-dir", VariantType::String, "/dev/null", {"metadata output directory, must exist (if not /dev/null)"}},
                             {"autosave-interval", VariantType::Int, 0, {"Write output to file for every n-th TF. 0 means this feature is OFF"}}}};
}

} // end namespace trd
} // end namespace o2

#endif // O2_TRD_KRWRITERSPEC_H
