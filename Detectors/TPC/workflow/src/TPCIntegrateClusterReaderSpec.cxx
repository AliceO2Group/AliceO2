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

/// @file   TPCIntegrateClusterReaderSpec.cxx

#include <vector>
#include <boost/algorithm/string/predicate.hpp>

#include "Framework/DeviceSpec.h"
#include "TPCWorkflow/TPCIntegrateClusterReaderSpec.h"
#include "DetectorsCalibration/IntegratedClusterCalibrator.h"
#include "TPCWorkflow/TPCIntegrateClusterSpec.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "Algorithm/RangeTokenizer.h"
#include "TChain.h"
#include "TGrid.h"
#include "TChainElement.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class IntegratedClusterReader : public Task
{
 public:
  IntegratedClusterReader() = default;
  ~IntegratedClusterReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTrees();
  void fillIndices();
  void fillIndicesFromMeta();

  /// create a TTree containing the meta informations from all the input files (can be used as input file to speed up the reading)
  void createMetaTTree(ProcessingContext& pc) const;

  int mLane = 0;
  int mNLanes = 1;
  int mChainEntry = 0;                                       ///< processed entries in the chain
  int mFirstTF{0};                                           ///< first TF to process
  int mLastTF{-1};                                           ///< last TF to process
  std::unique_ptr<TChain> mChain;                            ///< input TChain
  std::vector<std::string> mFileNames;                       ///< input files
  ITPCC mTPCC, *mTPCCPtr = &mTPCC;                           ///< branch integrated number of cluster TPC currents
  o2::dataformats::TFIDInfo mTFinfo, *mTFinfoPtr = &mTFinfo; ///< branch TFIDInfo for injecting correct time
  std::vector<std::tuple<unsigned long, int, int>> mIndices; ///< firstTfOrbit, file, index
  std::string mMetaOutFileDir{};                             ///< output dir for meta object (if empty no object will be created)
  std::vector<std::string> mMetaInFiles{};                   ///< input dir for meta objects (if empty no object will be loaded)
  std::unique_ptr<TFile> mTFile;                             ///< TFile in case meta data is used for speed up accessing files
  TTree* mTree = nullptr;                                    ///< TTree associated to the mTFile
};

void IntegratedClusterReader::init(InitContext& ic)
{
  // possible output meta dir
  mMetaOutFileDir = ic.options().get<std::string>("output-meta-dir");
  if (mMetaOutFileDir != "none") {
    LOGP(info, "Setting up meta output directory to {}", mMetaOutFileDir);
    mMetaOutFileDir = o2::utils::Str::rectifyDirectory(mMetaOutFileDir);
  }

  mMetaInFiles = o2::RangeTokenizer::tokenize<std::string>(ic.options().get<std::string>("input-meta-files"));
  if (!mMetaInFiles.empty()) {
    // only one directory should be set!
    if (mMetaOutFileDir != "none") {
      LOGP(error, "Only input-meta-files or output-meta-dir should be set. Setting input-meta-files to none");
      mMetaInFiles.clear();
    } else {
      LOGP(info, "Setting up meta input files");
      if (mMetaInFiles.size() == 1) {
        if (boost::algorithm::ends_with(mMetaInFiles.front(), "txt")) {
          LOGP(info, "Reading meta files from input file {}", mMetaInFiles.front());
          std::ifstream is(mMetaInFiles.front());
          std::istream_iterator<std::string> start(is);
          std::istream_iterator<std::string> end;
          std::vector<std::string> fileNamesTmp(start, end);
          mMetaInFiles = fileNamesTmp;
        }
      }
    }
  }

  mFirstTF = ic.options().get<int>("firstTF");
  mLastTF = ic.options().get<int>("lastTF");
  mLane = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  mChainEntry = mLane;
  mNLanes = ic.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;

  const auto dontCheckFileAccess = ic.options().get<bool>("dont-check-file-access");
  auto fileList = o2::RangeTokenizer::tokenize<std::string>(ic.options().get<std::string>("tpc-currents-infiles"));

  // check if only one input file (a txt file contaning a list of files is provided)
  if (fileList.size() == 1) {
    if (boost::algorithm::ends_with(fileList.front(), "txt")) {
      LOGP(info, "Reading files from input file {}", fileList.front());
      std::ifstream is(fileList.front());
      std::istream_iterator<std::string> start(is);
      std::istream_iterator<std::string> end;
      std::vector<std::string> fileNamesTmp(start, end);
      fileList = fileNamesTmp;
    }
  }

  const std::string inpDir = o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir"));
  for (const auto& file : fileList) {
    if ((file.find("alien://") == 0) && !gGrid && !TGrid::Connect("alien://")) {
      LOG(fatal) << "Failed to open alien connection";
    }
    const auto fileDir = o2::utils::Str::concat_string(inpDir, file);
    if (!dontCheckFileAccess) {
      std::unique_ptr<TFile> filePtr(TFile::Open(fileDir.data()));
      if (!filePtr || !filePtr->IsOpen() || filePtr->IsZombie()) {
        LOGP(warning, "Could not open file {}", fileDir);
        continue;
      }
    }
    mFileNames.emplace_back(fileDir);
  }

  if (mFileNames.size() == 0) {
    LOGP(error, "No input files to process");
  }
  connectTrees();
}

void IntegratedClusterReader::run(ProcessingContext& pc)
{
  // check time order inside the TChain
  if (mChainEntry == mLane) {
    // create meta data if requested
    if (mMetaOutFileDir != "none") {
      createMetaTTree(pc);
      LOGP(info, "Quit");
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      return;
    }

    if (!mMetaInFiles.empty()) {
      LOGP(info, "Reading in meta data from files...");
      fillIndicesFromMeta();
    } else {
      fillIndices();
    }
    LOGP(info, "Processing {} TFs out ouf {} TFs with first index in chain {}", mLastTF, mIndices.size(), mChainEntry);
  }

  if (mChainEntry >= mIndices.size() || (mLastTF != -1 && (pc.services().get<o2::framework::TimingInfo>().tfCounter >= mLastTF))) {
    LOGP(info, "Quit. mChainEntry {}  mIndices.size {} mLastTF {} tfCounter {}", mChainEntry, mIndices.size(), mLastTF, pc.services().get<o2::framework::TimingInfo>().tfCounter);
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  if (mMetaInFiles.empty()) {
    const int entry = std::get<1>(mIndices[mChainEntry]);
    LOGP(info, "Processing entry {}", entry);
    mChain->GetEntry(entry);
  } else {
    const int iFile = std::get<1>(mIndices[mChainEntry]);
    const int treeEntry = std::get<2>(mIndices[mChainEntry]);
    LOGP(info, "Processing file {} with TTree index {}", iFile, treeEntry);
    TChainElement* chainEle = (TChainElement*)(mChain->GetListOfFiles()->At(iFile));

    // check if last file is sane as current file
    if (mTFile && (std::string(mTFile->GetName()) == std::string(chainEle->GetTitle()))) {
      LOGP(info, "File is same. Do not need to reload...");
    } else {
      // open new file, destroy old TTree
      mTFile = std::unique_ptr<TFile>{TFile::Open(chainEle->GetTitle())};
      mTree = nullptr;
    }

    if (!mTFile) {
      LOGP(warning, "File {} is nullptr", chainEle->GetTitle());
      return;
    }

    if (!mTree) {
      LOGP(info, "Setting up new TTree");
      mTFile->GetObject("itpcc", mTree);
      if (!mTree) {
        LOGP(warning, "Tree for file {} is nullptr", chainEle->GetTitle());
        return;
      }

      mTree->SetBranchAddress("ITPCC", &mTPCCPtr);
      mTree->SetBranchAddress("tfID", &mTFinfoPtr);
    }
    mTree->GetEntry(treeEntry);
  }
  mChainEntry += mNLanes;

  // inject correct timing informations
  auto& timingInfo = pc.services().get<o2::framework::TimingInfo>();
  timingInfo.firstTForbit = mTFinfo.firstTForbit;
  timingInfo.tfCounter = mTFinfo.tfCounter;
  timingInfo.runNumber = mTFinfo.runNumber;
  timingInfo.creation = mTFinfo.creation;

  LOGP(info, "Processed data for firstTForbit {} and tfCounter {}", timingInfo.firstTForbit, timingInfo.tfCounter);
  pc.outputs().snapshot(Output{header::gDataOriginTPC, getDataDescriptionTPCC()}, mTPCC);
  usleep(100);
}

void IntegratedClusterReader::connectTrees()
{
  mChain.reset(new TChain("itpcc"));
  for (const auto& file : mFileNames) {
    LOGP(info, "Adding file to chain: {}", file);
    mChain->AddFile(file.data());
  }
  assert(mChain->GetEntries());
  mChain->SetBranchAddress("ITPCC", &mTPCCPtr);
  mChain->SetBranchAddress("tfID", &mTFinfoPtr);
}

void IntegratedClusterReader::createMetaTTree(ProcessingContext& pc) const
{
  const std::string outFileMeta = fmt::format("{}/tpc_meta_{}.root", mMetaOutFileDir, mLane);
  LOGP(info, "Producing meta file to: {}", outFileMeta);

  const int nFilesInChain = mChain->GetListOfFiles()->GetEntries();
  const int nFilesPerLane = nFilesInChain / mNLanes;
  const int firstFile = mLane * nFilesPerLane;
  const int lastFile = (mLane == mNLanes - 1) ? nFilesInChain : (firstFile + nFilesPerLane);
  LOGP(info, "Processing files {} to {}", firstFile, lastFile - 1);

  utils::TreeStreamRedirector pcstream(outFileMeta.data(), "RECREATE");
  for (unsigned int iFile = firstFile; iFile < lastFile; ++iFile) {
    TChainElement* chainEle = (TChainElement*)(mChain->GetListOfFiles()->At(iFile));
    auto file = std::unique_ptr<TFile>{TFile::Open(chainEle->GetTitle())};
    if (!file) {
      LOGP(warning, "File {} is nullptr", chainEle->GetTitle());
      continue;
    }
    TTree* tree = nullptr;
    file->GetObject("itpcc", tree);
    if (!tree) {
      LOGP(warning, "Tree for file {} is nullptr", chainEle->GetTitle());
      continue;
    }
    unsigned int firstTForbit = 0;
    unsigned int tfCounter = 0;
    tree->SetBranchAddress("tfID.firstTForbit", &firstTForbit);
    tree->SetBranchAddress("tfID.tfCounter", &tfCounter);
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("firstTForbit", 1);
    tree->SetBranchStatus("tfCounter", 1);
    const int entries = tree->GetEntries();
    for (unsigned int i = 0; i < entries; ++i) {
      tree->GetEntry(i);
      pcstream << "meta"
               << "firstTForbit=" << firstTForbit
               << "tfCounter=" << tfCounter // TF id
               << "tree_entry=" << i        // entry in ttree
               << "file=" << iFile          // file in chain
               << "lane=" << mLane          // current lane
               << "\n";
    }
  }
  LOGP(info, "Created meta data for input files {} to {} for a total of {} files", firstFile, lastFile - 1, nFilesInChain);
}

void IntegratedClusterReader::fillIndices()
{
  LOGP(info, "Filling indices from input files");
  mIndices.clear();
  mIndices.reserve(mChain->GetEntries());
  // disable all branches except the firstTForbit branch to significantly speed up the loop over the TTree
  mChain->SetBranchStatus("*", 0);
  mChain->SetBranchStatus("firstTForbit", 1);
  uint32_t countTFs = mChain->GetEntries();
  // in case processing doesnt start from the first TF, the first TF to be processed needs to be defined
  if (mLastTF != -1) {
    countTFs = 0;
    mChain->SetBranchStatus("tfCounter", 1);
  }
  for (unsigned long i = 0; i < mChain->GetEntries(); i++) {
    mChain->GetEntry(i);
    // in case indices are loaded directly from the chain store the global index
    mIndices.emplace_back(std::make_tuple(mTFinfo.firstTForbit, i, -1));

    if ((mLastTF != -1) && (mTFinfo.tfCounter <= mLastTF)) {
      if ((mTFinfo.tfCounter >= mFirstTF)) {
        // count number of TFs to process
        ++countTFs;
      } else {
        // keep track of first entry in the chain
        ++mChainEntry;
      }
    }
  }
  mChain->SetBranchStatus("*", 1);
  std::sort(mIndices.begin(), mIndices.end());
  mLastTF = countTFs;
}

void IntegratedClusterReader::fillIndicesFromMeta()
{
  std::vector<int> indicesMeta(mMetaInFiles.size()); // sorted indices if input meta files are not sorted
  for (int ifile = 0; ifile < mMetaInFiles.size(); ++ifile) {
    auto file = std::unique_ptr<TFile>{TFile::Open(mMetaInFiles[ifile].data())};
    TTree* tree = nullptr;
    file->GetObject("meta", tree);
    if (!tree) {
      LOGP(warning, "Tree for file {} is nullptr", mMetaInFiles[ifile]);
      continue;
    }
    int fileIdx = 0;
    tree->SetBranchAddress("lane", &fileIdx);
    tree->GetEntry(0);
    LOGP(info, "Index of file {} is {}", ifile, fileIdx);
    indicesMeta[fileIdx] = ifile;
  }
  TChain metaChain("meta");
  for (const auto idxFile : indicesMeta) {
    LOGP(info, "Adding to meta chain: {}", mMetaInFiles[idxFile]);
    metaChain.Add(mMetaInFiles[idxFile].data());
  }

  assert(metaChain.GetEntries());

  unsigned int firstTForbit = 0;
  unsigned int tfCounter = 0;
  unsigned int tree_entry = 0;
  unsigned int file = 0;
  metaChain.SetBranchAddress("firstTForbit", &firstTForbit);
  metaChain.SetBranchAddress("tfCounter", &tfCounter);
  metaChain.SetBranchAddress("tree_entry", &tree_entry);
  metaChain.SetBranchAddress("file", &file);

  LOGP(info, "Filling indices from local meta files");
  mIndices.clear();
  mIndices.reserve(metaChain.GetEntries());

  uint32_t countTFs = (mLastTF != -1) ? 0 : metaChain.GetEntries();

  // storage of relevant indices per file
  std::unordered_map<int, std::vector<std::tuple<unsigned long, int, int>>> indices_per_file;

  // in case processing doesnt start from the first TF, the first TF to be processed needs to be defined
  for (unsigned long i = 0; i < metaChain.GetEntries(); i++) {
    metaChain.GetEntry(i);
    // in case indices are loaded  from the meta file store the file index and ttree index
    mIndices.emplace_back(std::make_tuple(firstTForbit, file, tree_entry));
    if ((mLastTF != -1) && (tfCounter <= mLastTF)) {
      if ((tfCounter >= mFirstTF)) {
        ++countTFs;
        indices_per_file[file].emplace_back(firstTForbit, file, tree_entry);
      } else {
        ++mChainEntry;
      }
    }
  }
  std::sort(mIndices.begin(), mIndices.end());
  mLastTF = countTFs;

  if (!indices_per_file.empty()) {
    mChainEntry = mIndices.size();
    int counterFile = 0;
    for (auto& indices : indices_per_file) {
      if ((mLane + counterFile++) % mNLanes) {
        continue;
      }
      auto& vIndices = indices.second;
      std::sort(vIndices.begin(), vIndices.end());
      const int firstIdx = mIndices.size();
      const int maxSize = vIndices.size() * mNLanes + firstIdx;
      mIndices.resize(maxSize); // extend vector if necessary and set extended values to -1
      LOGP(info, "Adding file {} to indices by increasing max size to {}", counterFile - 1, maxSize);
      for (int i = 0; i < vIndices.size(); ++i) {
        mIndices[firstIdx + i * mNLanes] = vIndices[i];
      }
    }
  }
}

DataProcessorSpec getTPCIntegrateClusterReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTPC, getDataDescriptionTPCC(), 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-integrated-cluster-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<IntegratedClusterReader>()},
    Options{
      {"tpc-currents-infiles", VariantType::String, "o2currents_tpc.root", {"comma-separated list of input files or .txt file containing list of input files"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"dont-check-file-access", VariantType::Bool, false, {"Deactivate check if all files are accessible before adding them to the list of files"}},
      {"firstTF", VariantType::Int, 0, {"First TF to process"}},
      {"lastTF", VariantType::Int, -1, {"Last TF to process"}},
      {"input-meta-files", VariantType::String, "", {"Input directory for meta data (TTree containing firstTForbit and tfCounter)"}},
      {"output-meta-dir", VariantType::String, "none", {"Output directory (TTree containing firstTForbit and tfCounter)"}},
    }};
}

} // namespace tpc
} // namespace o2
