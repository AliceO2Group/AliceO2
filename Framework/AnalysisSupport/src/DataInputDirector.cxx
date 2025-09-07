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
#include "DataInputDirector.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/Logger.h"
#include "Framework/PluginManager.h"
#include "Framework/RootArrowFilesystem.h"
#include "Framework/AnalysisDataModelHelpers.h"
#include "Framework/Output.h"
#include "Framework/Signpost.h"
#include "Headers/DataHeader.h"
#include "Framework/TableTreeHelpers.h"
#include "Monitoring/Tags.h"
#include "Monitoring/Metric.h"
#include "Monitoring/Monitoring.h"

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"

#include "TGrid.h"
#include "TObjString.h"
#include "TMap.h"
#include "TFile.h"

#include <arrow/dataset/file_base.h>
#include <arrow/dataset/dataset.h>
#include <uv.h>
#include <memory>

#if __has_include(<TJAlienFile.h>)
#include <TJAlienFile.h>

#include <utility>
#endif

#include <dlfcn.h>
O2_DECLARE_DYNAMIC_LOG(reader_memory_dump);

namespace o2::framework
{
using namespace rapidjson;

FileNameHolder* makeFileNameHolder(std::string fileName)
{
  auto fileNameHolder = new FileNameHolder();
  fileNameHolder->fileName = fileName;

  return fileNameHolder;
}

DataInputDescriptor::DataInputDescriptor(bool alienSupport, int level, o2::monitoring::Monitoring* monitoring, int allowedParentLevel, std::string parentFileReplacement)
  : mAlienSupport(alienSupport),
    mMonitoring(monitoring),
    mAllowedParentLevel(allowedParentLevel),
    mParentFileReplacement(std::move(parentFileReplacement)),
    mLevel(level)
{
  std::vector<char const*> capabilitiesSpecs = {
    "O2Framework:RNTupleObjectReadingCapability",
    "O2Framework:TTreeObjectReadingCapability",
  };

  std::vector<LoadablePlugin> plugins;
  for (auto spec : capabilitiesSpecs) {
    auto morePlugins = PluginManager::parsePluginSpecString(spec);
    for (auto& extra : morePlugins) {
      plugins.push_back(extra);
    }
  }

  PluginManager::loadFromPlugin<RootObjectReadingCapability, RootObjectReadingCapabilityPlugin>(plugins, mFactory.capabilities);
}

void DataInputDescriptor::printOut()
{
  LOGP(info, "DataInputDescriptor");
  LOGP(info, "  Table name        : {}", tablename);
  LOGP(info, "  Tree name         : {}", treename);
  LOGP(info, "  Input files file  : {}", getInputfilesFilename());
  LOGP(info, "  File name regex   : {}", getFilenamesRegexString());
  LOGP(info, "  Input files       : {}", mfilenames.size());
  for (auto fn : mfilenames) {
    LOGP(info, "    {} {}", fn->fileName, fn->numberOfTimeFrames);
  }
  LOGP(info, "  Total number of TF: {}", getNumberTimeFrames());
}

std::string DataInputDescriptor::getInputfilesFilename()
{
  return (minputfilesFile.empty() && minputfilesFilePtr) ? (std::string)*minputfilesFilePtr : minputfilesFile;
}

std::string DataInputDescriptor::getFilenamesRegexString()
{
  return (mFilenameRegex.empty() && mFilenameRegexPtr) ? (std::string)*mFilenameRegexPtr : mFilenameRegex;
}

std::regex DataInputDescriptor::getFilenamesRegex()
{
  return std::regex(getFilenamesRegexString());
}

void DataInputDescriptor::addFileNameHolder(FileNameHolder* fn)
{
  // remove leading file:// from file name
  if (fn->fileName.rfind("file://", 0) == 0) {
    fn->fileName.erase(0, 7);
  } else if (!mAlienSupport && fn->fileName.rfind("alien://", 0) == 0 && !gGrid) {
    LOGP(debug, "AliEn file requested. Enabling support.");
    TGrid::Connect("alien://");
    mAlienSupport = true;
  }

  mtotalNumberTimeFrames += fn->numberOfTimeFrames;
  mfilenames.emplace_back(fn);
}

bool DataInputDescriptor::setFile(int counter, std::string_view origin)
{
  // no files left
  if (counter >= getNumberInputfiles()) {
    return false;
  }

  // In case the origin starts with a anything but AOD, we add the origin as the suffix
  // of the filename. In the future we might expand this for proper rewriting of the
  // filename based on the origin and the original file information.
  std::string filename = mfilenames[counter]->fileName;
  if (!origin.starts_with("AOD")) {
    filename = std::regex_replace(filename, std::regex("[.]root$"), fmt::format("_{}.root", origin));
  }

  // open file
  auto rootFS = std::dynamic_pointer_cast<TFileFileSystem>(mCurrentFilesystem);
  if (rootFS.get()) {
    if (rootFS->GetFile()->GetName() == filename) {
      return true;
    }
    closeInputFile();
  }

  mCurrentFilesystem = std::make_shared<TFileFileSystem>(TFile::Open(filename.c_str()), 50 * 1024 * 1024, mFactory);
  if (!mCurrentFilesystem.get()) {
    throw std::runtime_error(fmt::format("Couldn't open file \"{}\"!", filename));
  }
  rootFS = std::dynamic_pointer_cast<TFileFileSystem>(mCurrentFilesystem);
  printFileOpening();

  // get the parent file map if exists
  mParentFileMap = (TMap*)rootFS->GetFile()->Get("parentFiles"); // folder name (DF_XXX) --> parent file (absolute path)
  if (mParentFileMap && !mParentFileReplacement.empty()) {
    auto pos = mParentFileReplacement.find(';');
    if (pos == std::string::npos) {
      throw std::runtime_error(fmt::format("Invalid syntax in aod-parent-base-path-replacement: \"{}\"", mParentFileReplacement.c_str()));
    }
    auto from = mParentFileReplacement.substr(0, pos);
    auto to = mParentFileReplacement.substr(pos + 1);

    auto it = mParentFileMap->MakeIterator();
    while (auto obj = it->Next()) {
      auto objString = (TObjString*)mParentFileMap->GetValue(obj);
      objString->String().ReplaceAll(from.c_str(), to.c_str());
    }
    delete it;
  }

  // get the directory names
  if (mfilenames[counter]->numberOfTimeFrames <= 0) {
    const std::regex TFRegex = std::regex("/?DF_([0-9]+)(|-.*)$");
    TList* keyList = rootFS->GetFile()->GetListOfKeys();
    std::vector<std::string> finalList;

    // extract TF numbers and sort accordingly
    // We use an extra seen set to make sure we preserve the order in which
    // we instert things in the final list and to make sure we do not have duplicates.
    // Multiple folder numbers can happen if we use a flat structure /DF_<df>-<tablename>
    std::unordered_set<size_t> seen;
    for (auto key : *keyList) {
      std::smatch matchResult;
      std::string keyName = ((TObjString*)key)->GetString().Data();
      bool match = std::regex_match(keyName, matchResult, TFRegex);
      if (match) {
        auto folderNumber = std::stoul(matchResult[1].str());
        if (seen.find(folderNumber) == seen.end()) {
          seen.insert(folderNumber);
          mfilenames[counter]->listOfTimeFrameNumbers.emplace_back(folderNumber);
        }
      }
    }

    if (mParentFileMap != nullptr) {
      // If we have a parent map, we should not process in DF alphabetical order but according to parent file to avoid swapping between files
      std::sort(mfilenames[counter]->listOfTimeFrameNumbers.begin(), mfilenames[counter]->listOfTimeFrameNumbers.end(),
                [this](long const& l1, long const& l2) -> bool {
                  auto p1 = (TObjString*)this->mParentFileMap->GetValue(("DF_" + std::to_string(l1)).c_str());
                  auto p2 = (TObjString*)this->mParentFileMap->GetValue(("DF_" + std::to_string(l2)).c_str());
                  return p1->GetString().CompareTo(p2->GetString()) < 0;
                });
    } else {
      std::sort(mfilenames[counter]->listOfTimeFrameNumbers.begin(), mfilenames[counter]->listOfTimeFrameNumbers.end());
    }

    mfilenames[counter]->alreadyRead.resize(mfilenames[counter]->alreadyRead.size() + mfilenames[counter]->listOfTimeFrameNumbers.size(), false);
    mfilenames[counter]->numberOfTimeFrames = mfilenames[counter]->listOfTimeFrameNumbers.size();
  }

  mCurrentFileID = counter;
  mCurrentFileStartedAt = uv_hrtime();
  mIOTime = 0;

  return true;
}

uint64_t DataInputDescriptor::getTimeFrameNumber(int counter, int numTF, std::string_view origin)
{

  // open file
  if (!setFile(counter, origin)) {
    return 0ul;
  }

  // no TF left
  if (mfilenames[counter]->numberOfTimeFrames > 0 && numTF >= mfilenames[counter]->numberOfTimeFrames) {
    return 0ul;
  }

  return (mfilenames[counter]->listOfTimeFrameNumbers)[numTF];
}

arrow::dataset::FileSource DataInputDescriptor::getFileFolder(int counter, int numTF, std::string_view origin)
{
  // open file
  if (!setFile(counter, origin)) {
    return {};
  }

  // no TF left
  if (mfilenames[counter]->numberOfTimeFrames > 0 && numTF >= mfilenames[counter]->numberOfTimeFrames) {
    return {};
  }

  mfilenames[counter]->alreadyRead[numTF] = true;

  return {fmt::format("DF_{}", mfilenames[counter]->listOfTimeFrameNumbers[numTF]), mCurrentFilesystem};
}

DataInputDescriptor* DataInputDescriptor::getParentFile(int counter, int numTF, std::string treename, std::string_view origin)
{
  if (!mParentFileMap) {
    // This file has no parent map
    return nullptr;
  }
  auto folderName = fmt::format("DF_{}", mfilenames[counter]->listOfTimeFrameNumbers[numTF]);
  auto parentFileName = (TObjString*)mParentFileMap->GetValue(folderName.c_str());
  // The current DF is not found in the parent map (this should not happen and is a fatal error)
  auto rootFS = std::dynamic_pointer_cast<TFileFileSystem>(mCurrentFilesystem);
  if (!parentFileName) {
    throw std::runtime_error(fmt::format(R"(parent file map exists but does not contain the current DF "{}" in file "{}")", folderName.c_str(), rootFS->GetFile()->GetName()));
    return nullptr;
  }

  if (mParentFile) {
    // Is this still the corresponding to the correct file?
    auto parentRootFS = std::dynamic_pointer_cast<TFileFileSystem>(mParentFile->mCurrentFilesystem);
    if (parentFileName->GetString().CompareTo(parentRootFS->GetFile()->GetName()) == 0) {
      return mParentFile;
    } else {
      mParentFile->closeInputFile();
      delete mParentFile;
      mParentFile = nullptr;
    }
  }

  if (mLevel == mAllowedParentLevel) {
    throw std::runtime_error(fmt::format(R"(while looking for tree "{}", the parent file was requested but we are already at level {} of maximal allowed level {} for DF "{}" in file "{}")", treename.c_str(), mLevel, mAllowedParentLevel, folderName.c_str(),
                                         rootFS->GetFile()->GetName()));
  }

  LOGP(info, "Opening parent file {} for DF {}", parentFileName->GetString().Data(), folderName.c_str());
  mParentFile = new DataInputDescriptor(mAlienSupport, mLevel + 1, mMonitoring, mAllowedParentLevel, mParentFileReplacement);
  mParentFile->mdefaultFilenamesPtr = new std::vector<FileNameHolder*>;
  mParentFile->mdefaultFilenamesPtr->emplace_back(makeFileNameHolder(parentFileName->GetString().Data()));
  mParentFile->fillInputfiles();
  mParentFile->setFile(0, origin);
  return mParentFile;
}

int DataInputDescriptor::getTimeFramesInFile(int counter)
{
  return mfilenames.at(counter)->numberOfTimeFrames;
}

int DataInputDescriptor::getReadTimeFramesInFile(int counter)
{
  auto& list = mfilenames.at(counter)->alreadyRead;
  return std::count(list.begin(), list.end(), true);
}

void DataInputDescriptor::printFileOpening()
{
  auto rootFS = std::dynamic_pointer_cast<TFileFileSystem>(mCurrentFilesystem);
  auto f = dynamic_cast<TFile*>(rootFS->GetFile());
  std::string monitoringInfo(fmt::format("lfn={},size={}", f->GetName(), f->GetSize()));
#if __has_include(<TJAlienFile.h>)
  auto alienFile = dynamic_cast<TJAlienFile*>(f);
  if (alienFile) {
    monitoringInfo += fmt::format(",se={},open_time={:.1f}", alienFile->GetSE(), alienFile->GetElapsed());
  }
#endif
  mMonitoring->send(o2::monitoring::Metric{monitoringInfo, "aod-file-open-info"}.addTag(o2::monitoring::tags::Key::Subsystem, o2::monitoring::tags::Value::DPL));
  LOGP(info, "Opening file: {}", monitoringInfo);
}

void DataInputDescriptor::printFileStatistics()
{
  int64_t wait_time = (int64_t)uv_hrtime() - (int64_t)mCurrentFileStartedAt - (int64_t)mIOTime;
  if (wait_time < 0) {
    wait_time = 0;
  }
  auto rootFS = std::dynamic_pointer_cast<TFileFileSystem>(mCurrentFilesystem);
  auto f = dynamic_cast<TFile*>(rootFS->GetFile());
  std::string monitoringInfo(fmt::format("lfn={},size={},total_df={},read_df={},read_bytes={},read_calls={},io_time={:.1f},wait_time={:.1f},level={}", f->GetName(),
                                         f->GetSize(), getTimeFramesInFile(mCurrentFileID), getReadTimeFramesInFile(mCurrentFileID), f->GetBytesRead(), f->GetReadCalls(),
                                         ((float)mIOTime / 1e9), ((float)wait_time / 1e9), mLevel));
#if __has_include(<TJAlienFile.h>)
  auto alienFile = dynamic_cast<TJAlienFile*>(f);
  if (alienFile) {
    monitoringInfo += fmt::format(",se={},open_time={:.1f}", alienFile->GetSE(), alienFile->GetElapsed());
  }
#endif
  mMonitoring->send(o2::monitoring::Metric{monitoringInfo, "aod-file-read-info"}.addTag(o2::monitoring::tags::Key::Subsystem, o2::monitoring::tags::Value::DPL));
  LOGP(info, "Read info: {}", monitoringInfo);
}

void DataInputDescriptor::closeInputFile()
{
  if (mCurrentFilesystem.get()) {
    if (mParentFile) {
      mParentFile->closeInputFile();
      delete mParentFile;
      mParentFile = nullptr;
    }

    delete mParentFileMap;
    mParentFileMap = nullptr;

    printFileStatistics();
    mCurrentFilesystem.reset();
  }
}

int DataInputDescriptor::fillInputfiles()
{
  if (getNumberInputfiles() > 0) {
    // 1. mfilenames
    return getNumberInputfiles();
  }

  auto fileName = getInputfilesFilename();
  if (!fileName.empty()) {
    // 2. getFilenamesRegex() @ getInputfilesFilename()
    try {
      std::ifstream filelist(fileName);
      if (!filelist.is_open()) {
        throw std::runtime_error(fmt::format(R"(Couldn't open file "{}")", fileName));
      }
      while (std::getline(filelist, fileName)) {
        // remove white spaces, empty lines are skipped
        fileName.erase(std::remove_if(fileName.begin(), fileName.end(), ::isspace), fileName.end());
        if (!fileName.empty() && (getFilenamesRegexString().empty() ||
                                  std::regex_match(fileName, getFilenamesRegex()))) {
          addFileNameHolder(makeFileNameHolder(fileName));
        }
      }
    } catch (...) {
      LOGP(error, "Check the input files file! Unable to process \"{}\"!", getInputfilesFilename());
      return 0;
    }
  } else {
    // 3. getFilenamesRegex() @ mdefaultFilenamesPtr
    if (mdefaultFilenamesPtr) {
      for (auto fileNameHolder : *mdefaultFilenamesPtr) {
        if (getFilenamesRegexString().empty() ||
            std::regex_match(fileNameHolder->fileName, getFilenamesRegex())) {
          addFileNameHolder(fileNameHolder);
        }
      }
    }
  }

  return getNumberInputfiles();
}

int DataInputDescriptor::findDFNumber(int file, std::string dfName)
{
  auto dfList = mfilenames[file]->listOfTimeFrameNumbers;
  auto it = std::find_if(dfList.begin(), dfList.end(), [dfName](size_t i) { return fmt::format("DF_{}", i) == dfName; });
  if (it == dfList.end()) {
    return -1;
  }
  return it - dfList.begin();
}

struct CalculateDelta {
  CalculateDelta(uint64_t& target)
    : mTarget(target)
  {
    start = uv_hrtime();
  }
  ~CalculateDelta()
  {
    if (!active) {
      return;
    }
    O2_SIGNPOST_ACTION(reader_memory_dump, [](void*) {
      void (*dump_)(const char*);
      if (void* sym = dlsym(nullptr, "igprof_dump_now")) {
        dump_ = __extension__(void (*)(const char*)) sym;
        if (dump_) {
          std::string filename = fmt::format("reader-memory-dump-{}.gz", uv_hrtime());
          dump_(filename.c_str());
        }
      }
    });
    mTarget += (uv_hrtime() - start);
  }

  void deactivate()
  {
    active = false;
  }

  bool active = true;
  uint64_t& mTarget;
  uint64_t start;
  uint64_t stop;
};

bool DataInputDescriptor::readTree(DataAllocator& outputs, header::DataHeader dh, int counter, int numTF, std::string treename, size_t& totalSizeCompressed, size_t& totalSizeUncompressed)
{
  CalculateDelta t(mIOTime);
  std::string origin = dh.dataOrigin.as<std::string>();
  auto folder = getFileFolder(counter, numTF, origin);
  if (!folder.filesystem()) {
    t.deactivate();
    return false;
  }

  auto rootFS = std::dynamic_pointer_cast<TFileFileSystem>(folder.filesystem());

  if (!rootFS) {
    t.deactivate();
    throw std::runtime_error(fmt::format(R"(Not a TFile filesystem!)"));
  }
  // FIXME: Ugly. We should detect the format from the treename, good enough for now.
  std::shared_ptr<arrow::dataset::FileFormat> format;
  FragmentToBatch::StreamerCreator creator = nullptr;

  auto fullpath = arrow::dataset::FileSource{folder.path() + "/" + treename, folder.filesystem()};

  for (auto& capability : mFactory.capabilities) {
    auto objectPath = capability.lfn2objectPath(fullpath.path());
    void* handle = capability.getHandle(rootFS, objectPath);
    if (handle) {
      format = capability.factory().format();
      creator = capability.factory().deferredOutputStreamer;
      break;
    }
  }

  // FIXME: we should distinguish between an actually missing object and one which has a non compatible
  // format.
  if (!format) {
    t.deactivate();
    LOGP(debug, "Could not find tree {}. Trying in parent file.", fullpath.path());
    auto parentFile = getParentFile(counter, numTF, treename, origin);
    if (parentFile != nullptr) {
      int parentNumTF = parentFile->findDFNumber(0, folder.path());
      if (parentNumTF == -1) {
        auto parentRootFS = std::dynamic_pointer_cast<TFileFileSystem>(parentFile->mCurrentFilesystem);
        throw std::runtime_error(fmt::format(R"(DF {} listed in parent file map but not found in the corresponding file "{}")", folder.path(), parentRootFS->GetFile()->GetName()));
      }
      // first argument is 0 as the parent file object contains only 1 file
      return parentFile->readTree(outputs, dh, 0, parentNumTF, treename, totalSizeCompressed, totalSizeUncompressed);
    }
    auto rootFS = std::dynamic_pointer_cast<TFileFileSystem>(mCurrentFilesystem);
    throw std::runtime_error(fmt::format(R"(Couldn't get TTree "{}" from "{}". Please check https://aliceo2group.github.io/analysis-framework/docs/troubleshooting/#tree-not-found for more information.)", fullpath.path(), rootFS->GetFile()->GetName()));
  }

  auto schemaOpt = format->Inspect(fullpath);
  auto physicalSchema = schemaOpt;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (auto& original : (*schemaOpt)->fields()) {
    if (original->name().ends_with("_size")) {
      continue;
    }
    fields.push_back(original);
  }
  auto datasetSchema = std::make_shared<arrow::Schema>(fields);

  auto fragment = format->MakeFragment(fullpath, {}, *physicalSchema);

  // create table output
  auto o = Output(dh);

  // FIXME: This should allow me to create a memory pool
  // which I can then use to scan the dataset.
  auto f2b = outputs.make<FragmentToBatch>(o, creator, *fragment);

  //// add branches to read
  //// fill the table
  f2b->setLabel(treename.c_str());
  f2b->fill(datasetSchema, format);

  return true;
}

DataInputDirector::DataInputDirector()
{
  createDefaultDataInputDescriptor();
}

DataInputDirector::DataInputDirector(std::string inputFile, o2::monitoring::Monitoring* monitoring, int allowedParentLevel, std::string parentFileReplacement) : mMonitoring(monitoring), mAllowedParentLevel(allowedParentLevel), mParentFileReplacement(std::move(parentFileReplacement))
{
  if (inputFile.size() && inputFile[0] == '@') {
    inputFile.erase(0, 1);
    setInputfilesFile(inputFile);
  } else {
    mdefaultInputFiles.emplace_back(makeFileNameHolder(inputFile));
  }

  createDefaultDataInputDescriptor();
}

DataInputDirector::DataInputDirector(std::vector<std::string> inputFiles, o2::monitoring::Monitoring* monitoring, int allowedParentLevel, std::string parentFileReplacement) : mMonitoring(monitoring), mAllowedParentLevel(allowedParentLevel), mParentFileReplacement(std::move(parentFileReplacement))
{
  for (auto inputFile : inputFiles) {
    mdefaultInputFiles.emplace_back(makeFileNameHolder(inputFile));
  }

  createDefaultDataInputDescriptor();
}

DataInputDirector::~DataInputDirector()
{
  for (auto fn : mdefaultInputFiles) {
    delete fn;
  }
  mdefaultInputFiles.clear();
  mdefaultDataInputDescriptor = nullptr;

  for (auto fn : mdataInputDescriptors) {
    delete fn;
  }
  mdataInputDescriptors.clear();
}

void DataInputDirector::reset()
{
  mdataInputDescriptors.clear();
  mdefaultInputFiles.clear();
  mFilenameRegex = std::string("");
};

void DataInputDirector::createDefaultDataInputDescriptor()
{
  if (mdefaultDataInputDescriptor) {
    delete mdefaultDataInputDescriptor;
  }
  mdefaultDataInputDescriptor = new DataInputDescriptor(mAlienSupport, 0, mMonitoring, mAllowedParentLevel, mParentFileReplacement);

  mdefaultDataInputDescriptor->setInputfilesFile(minputfilesFile);
  mdefaultDataInputDescriptor->setFilenamesRegex(mFilenameRegex);
  mdefaultDataInputDescriptor->setDefaultInputfiles(&mdefaultInputFiles);
  mdefaultDataInputDescriptor->tablename = "any";
  mdefaultDataInputDescriptor->treename = "any";
  mdefaultDataInputDescriptor->fillInputfiles();

  mAlienSupport &= mdefaultDataInputDescriptor->isAlienSupportOn();
}

bool DataInputDirector::readJson(std::string const& fnjson)
{
  // open the file
  FILE* f = fopen(fnjson.c_str(), "r");
  if (!f) {
    LOGP(error, "Could not open JSON file \"{}\"!", fnjson);
    return false;
  }

  // create streamer
  char readBuffer[65536];
  FileReadStream inputStream(f, readBuffer, sizeof(readBuffer));

  // parse the json file
  Document jsonDoc;
  jsonDoc.ParseStream(inputStream);
  auto status = readJsonDocument(&jsonDoc);

  // clean up
  fclose(f);

  return status;
}

bool DataInputDirector::readJsonDocument(Document* jsonDoc)
{
  // initialisations
  std::string fileName("");
  const char* itemName;

  // is it a proper json document?
  if (jsonDoc->HasParseError()) {
    LOGP(error, "Check the JSON document! There is a problem with the format!");
    return false;
  }

  // InputDirector
  itemName = "InputDirector";
  const Value& didirItem = (*jsonDoc)[itemName];
  if (!didirItem.IsObject()) {
    LOGP(info, "No \"{}\" object found in the JSON document!", itemName);
    return true;
  }

  // now read various items
  itemName = "debugmode";
  if (didirItem.HasMember(itemName)) {
    if (didirItem[itemName].IsBool()) {
      mDebugMode = (didirItem[itemName].GetBool());
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a boolean!", itemName);
      return false;
    }
  } else {
    mDebugMode = false;
  }

  if (mDebugMode) {
    StringBuffer buffer;
    buffer.Clear();
    PrettyWriter<StringBuffer> writer(buffer);
    didirItem.Accept(writer);
    LOGP(info, "InputDirector object: {}", std::string(buffer.GetString()));
  }

  itemName = "fileregex";
  if (didirItem.HasMember(itemName)) {
    if (didirItem[itemName].IsString()) {
      setFilenamesRegex(didirItem[itemName].GetString());
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a string!", itemName);
      return false;
    }
  }

  itemName = "resfiles";
  if (didirItem.HasMember(itemName)) {
    if (didirItem[itemName].IsString()) {
      fileName = didirItem[itemName].GetString();
      if (fileName.size() && fileName[0] == '@') {
        fileName.erase(0, 1);
        setInputfilesFile(fileName);
      } else {
        setInputfilesFile("");
        mdefaultInputFiles.emplace_back(makeFileNameHolder(fileName));
      }
    } else if (didirItem[itemName].IsArray()) {
      setInputfilesFile("");
      auto fns = didirItem[itemName].GetArray();
      for (auto& fn : fns) {
        mdefaultInputFiles.emplace_back(makeFileNameHolder(fn.GetString()));
      }
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a string or an array!", itemName);
      return false;
    }
  }

  itemName = "InputDescriptors";
  if (didirItem.HasMember(itemName)) {
    if (!didirItem[itemName].IsArray()) {
      LOGP(error, "Check the JSON document! Item \"{}\" must be an array!", itemName);
      return false;
    }

    // loop over DataInputDescriptors
    for (auto& didescItem : didirItem[itemName].GetArray()) {
      if (!didescItem.IsObject()) {
        LOGP(error, "Check the JSON document! \"{}\" must be objects!", itemName);
        return false;
      }
      // create a new dataInputDescriptor
      auto didesc = new DataInputDescriptor(mAlienSupport, 0, mMonitoring, mAllowedParentLevel, mParentFileReplacement);
      didesc->setDefaultInputfiles(&mdefaultInputFiles);

      itemName = "table";
      if (didescItem.HasMember(itemName)) {
        if (didescItem[itemName].IsString()) {
          didesc->tablename = didescItem[itemName].GetString();
          didesc->matcher = DataDescriptorQueryBuilder::buildNode(didesc->tablename);
        } else {
          LOGP(error, "Check the JSON document! Item \"{}\" must be a string!", itemName);
          return false;
        }
      } else {
        LOGP(error, "Check the JSON document! Item \"{}\" is missing!", itemName);
        return false;
      }

      itemName = "treename";
      if (didescItem.HasMember(itemName)) {
        if (didescItem[itemName].IsString()) {
          didesc->treename = didescItem[itemName].GetString();
        } else {
          LOGP(error, "Check the JSON document! Item \"{}\" must be a string!", itemName);
          return false;
        }
      } else {
        auto m = DataDescriptorQueryBuilder::getTokens(didesc->tablename);
        didesc->treename = m[2];
      }

      itemName = "fileregex";
      if (didescItem.HasMember(itemName)) {
        if (didescItem[itemName].IsString()) {
          if (didesc->getNumberInputfiles() == 0) {
            didesc->setFilenamesRegex(didescItem[itemName].GetString());
          }
        } else {
          LOGP(error, "Check the JSON document! Item \"{}\" must be a string!", itemName);
          return false;
        }
      } else {
        if (didesc->getNumberInputfiles() == 0) {
          didesc->setFilenamesRegex(mFilenameRegexPtr);
        }
      }

      itemName = "resfiles";
      if (didescItem.HasMember(itemName)) {
        if (didescItem[itemName].IsString()) {
          fileName = didescItem[itemName].GetString();
          if (fileName.size() && fileName[0] == '@') {
            didesc->setInputfilesFile(fileName.erase(0, 1));
          } else {
            if (didesc->getFilenamesRegexString().empty() ||
                std::regex_match(fileName, didesc->getFilenamesRegex())) {
              didesc->addFileNameHolder(makeFileNameHolder(fileName));
            }
          }
        } else if (didescItem[itemName].IsArray()) {
          auto fns = didescItem[itemName].GetArray();
          for (auto& fn : fns) {
            if (didesc->getFilenamesRegexString().empty() ||
                std::regex_match(fn.GetString(), didesc->getFilenamesRegex())) {
              didesc->addFileNameHolder(makeFileNameHolder(fn.GetString()));
            }
          }
        } else {
          LOGP(error, "Check the JSON document! Item \"{}\" must be a string or an array!", itemName);
          return false;
        }
      } else {
        didesc->setInputfilesFile(minputfilesFilePtr);
      }

      // fill mfilenames and add InputDescriptor to InputDirector
      if (didesc->fillInputfiles() > 0) {
        mdataInputDescriptors.emplace_back(didesc);
      } else {
        didesc->printOut();
        LOGP(info, "This DataInputDescriptor is ignored because its file list is empty!");
      }
      mAlienSupport &= didesc->isAlienSupportOn();
    }
  }

  // add a default DataInputDescriptor
  createDefaultDataInputDescriptor();

  // check that all DataInputDescriptors have the same number of input files
  if (!isValid()) {
    printOut();
    return false;
  }

  // print the DataIputDirector
  if (mDebugMode) {
    printOut();
  }

  return true;
}

DataInputDescriptor* DataInputDirector::getDataInputDescriptor(header::DataHeader dh)
{
  DataInputDescriptor* result = nullptr;

  // compute list of matching outputs
  data_matcher::VariableContext context;

  for (auto didesc : mdataInputDescriptors) {
    if (didesc->matcher->match(dh, context)) {
      result = didesc;
      break;
    }
  }

  return result;
}

arrow::dataset::FileSource DataInputDirector::getFileFolder(header::DataHeader dh, int counter, int numTF)
{
  auto didesc = getDataInputDescriptor(dh);
  // if NOT match then use defaultDataInputDescriptor
  if (!didesc) {
    didesc = mdefaultDataInputDescriptor;
  }
  std::string origin = dh.dataOrigin.as<std::string>();

  return didesc->getFileFolder(counter, numTF, origin);
}

int DataInputDirector::getTimeFramesInFile(header::DataHeader dh, int counter)
{
  auto didesc = getDataInputDescriptor(dh);
  // if NOT match then use defaultDataInputDescriptor
  if (!didesc) {
    didesc = mdefaultDataInputDescriptor;
  }

  return didesc->getTimeFramesInFile(counter);
}

uint64_t DataInputDirector::getTimeFrameNumber(header::DataHeader dh, int counter, int numTF)
{
  auto didesc = getDataInputDescriptor(dh);
  // if NOT match then use defaultDataInputDescriptor
  if (!didesc) {
    didesc = mdefaultDataInputDescriptor;
  }
  std::string origin = dh.dataOrigin.as<std::string>();

  return didesc->getTimeFrameNumber(counter, numTF, origin);
}

bool DataInputDirector::readTree(DataAllocator& outputs, header::DataHeader dh, int counter, int numTF, size_t& totalSizeCompressed, size_t& totalSizeUncompressed)
{
  std::string treename;

  auto didesc = getDataInputDescriptor(dh);
  if (didesc) {
    // if match then use filename and treename from DataInputDescriptor
    treename = didesc->treename;
  } else {
    // if NOT match then use
    //  . filename from defaultDataInputDescriptor
    //  . treename from DataHeader
    didesc = mdefaultDataInputDescriptor;
    treename = aod::datamodel::getTreeName(dh);
  }
  std::string origin = dh.dataOrigin.as<std::string>();

  auto result = didesc->readTree(outputs, dh, counter, numTF, treename, totalSizeCompressed, totalSizeUncompressed);
  return result;
}

void DataInputDirector::closeInputFiles()
{
  mdefaultDataInputDescriptor->closeInputFile();
  for (auto didesc : mdataInputDescriptors) {
    didesc->closeInputFile();
  }
}

bool DataInputDirector::isValid()
{
  bool status = true;
  int numberFiles = mdefaultDataInputDescriptor->getNumberInputfiles();
  for (auto didesc : mdataInputDescriptors) {
    status &= didesc->getNumberInputfiles() == numberFiles;
  }

  return status;
}

bool DataInputDirector::atEnd(int counter)
{
  bool status = mdefaultDataInputDescriptor->getNumberInputfiles() <= counter;
  for (auto didesc : mdataInputDescriptors) {
    status &= (didesc->getNumberInputfiles() <= counter);
  }

  return status;
}

void DataInputDirector::printOut()
{
  LOGP(info, "DataInputDirector");
  LOGP(info, "  Default input files file   : {}", minputfilesFile);
  LOGP(info, "  Default file name regex    : {}", mFilenameRegex);
  LOGP(info, "  Default file names         : {}", mdefaultInputFiles.size());
  for (auto const& fn : mdefaultInputFiles) {
    LOGP(info, "    {} {}", fn->fileName, fn->numberOfTimeFrames);
  }
  LOGP(info, "  Default DataInputDescriptor:");
  mdefaultDataInputDescriptor->printOut();
  LOGP(info, "  DataInputDescriptors       : {}", getNumberInputDescriptors());
  for (auto const& didesc : mdataInputDescriptors) {
    didesc->printOut();
  }
}

} // namespace o2::framework
