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
#include "Framework/AnalysisDataModelHelpers.h"
#include "Framework/Output.h"
#include "Headers/DataHeader.h"
#include "Framework/TableTreeHelpers.h"
#include "Monitoring/Tags.h"

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"

#include "TGrid.h"
#include "TObjString.h"
#include "TMap.h"

#include <uv.h>

#if __has_include(<TJAlienFile.h>)
#include <TJAlienFile.h>

#include <utility>
#endif

std::vector<std::string> getColumnNames(o2::header::DataHeader dh)
{
  auto description = std::string(dh.dataDescription.str);
  auto origin = std::string(dh.dataOrigin.str);

  // default: column names = {}
  return {};
}

namespace o2::framework
{
using namespace rapidjson;

FileNameHolder* makeFileNameHolder(std::string fileName)
{
  auto fileNameHolder = new FileNameHolder();
  fileNameHolder->fileName = fileName;

  return fileNameHolder;
}

DataInputDescriptor::DataInputDescriptor(bool alienSupport, int level, o2::monitoring::Monitoring* monitoring, int allowedParentLevel, std::string parentFileReplacement) : mAlienSupport(alienSupport),
                                                                                                                                                                            mMonitoring(monitoring),
                                                                                                                                                                            mAllowedParentLevel(allowedParentLevel),
                                                                                                                                                                            mParentFileReplacement(std::move(parentFileReplacement)),
                                                                                                                                                                            mLevel(level)
{
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
  } else if (!mAlienSupport && fn->fileName.rfind("alien://", 0) == 0) {
    LOGP(debug, "AliEn file requested. Enabling support.");
    TGrid::Connect("alien://");
    mAlienSupport = true;
  }

  mtotalNumberTimeFrames += fn->numberOfTimeFrames;
  mfilenames.emplace_back(fn);
}

bool DataInputDescriptor::setFile(int counter)
{
  // no files left
  if (counter >= getNumberInputfiles()) {
    return false;
  }

  // open file
  auto filename = mfilenames[counter]->fileName;
  if (mcurrentFile) {
    if (mcurrentFile->GetName() == filename) {
      return true;
    }
    closeInputFile();
  }
  mcurrentFile = TFile::Open(filename.c_str());
  if (!mcurrentFile) {
    throw std::runtime_error(fmt::format("Couldn't open file \"{}\"!", filename));
  }
  mcurrentFile->SetReadaheadSize(50 * 1024 * 1024);

  // get the parent file map if exists
  mParentFileMap = (TMap*)mcurrentFile->Get("parentFiles"); // folder name (DF_XXX) --> parent file (absolute path)
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
    std::regex TFRegex = std::regex("DF_[0-9]+");
    TList* keyList = mcurrentFile->GetListOfKeys();

    // extract TF numbers and sort accordingly
    for (auto key : *keyList) {
      if (std::regex_match(((TObjString*)key)->GetString().Data(), TFRegex)) {
        auto folderNumber = std::stoul(std::string(((TObjString*)key)->GetString().Data()).substr(3));
        mfilenames[counter]->listOfTimeFrameNumbers.emplace_back(folderNumber);
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

    for (auto folderNumber : mfilenames[counter]->listOfTimeFrameNumbers) {
      auto folderName = "DF_" + std::to_string(folderNumber);
      mfilenames[counter]->listOfTimeFrameKeys.emplace_back(folderName);
      mfilenames[counter]->alreadyRead.emplace_back(false);
    }
    mfilenames[counter]->numberOfTimeFrames = mfilenames[counter]->listOfTimeFrameKeys.size();
  }

  mCurrentFileID = counter;
  mCurrentFileStartedAt = uv_hrtime();

  return true;
}

uint64_t DataInputDescriptor::getTimeFrameNumber(int counter, int numTF)
{

  // open file
  if (!setFile(counter)) {
    return 0ul;
  }

  // no TF left
  if (mfilenames[counter]->numberOfTimeFrames > 0 && numTF >= mfilenames[counter]->numberOfTimeFrames) {
    return 0ul;
  }

  return (mfilenames[counter]->listOfTimeFrameNumbers)[numTF];
}

FileAndFolder DataInputDescriptor::getFileFolder(int counter, int numTF)
{
  FileAndFolder fileAndFolder;

  // open file
  if (!setFile(counter)) {
    return fileAndFolder;
  }

  // no TF left
  if (mfilenames[counter]->numberOfTimeFrames > 0 && numTF >= mfilenames[counter]->numberOfTimeFrames) {
    return fileAndFolder;
  }

  fileAndFolder.file = mcurrentFile;
  fileAndFolder.folderName = (mfilenames[counter]->listOfTimeFrameKeys)[numTF];

  mfilenames[counter]->alreadyRead[numTF] = true;

  return fileAndFolder;
}

DataInputDescriptor* DataInputDescriptor::getParentFile(int counter, int numTF)
{
  if (!mParentFileMap) {
    // This file has no parent map
    return nullptr;
  }
  auto folderName = (mfilenames[counter]->listOfTimeFrameKeys)[numTF];
  auto parentFileName = (TObjString*)mParentFileMap->GetValue(folderName.c_str());
  if (!parentFileName) {
    // The current DF is not found in the parent map (this should not happen and is a fatal error)
    throw std::runtime_error(fmt::format(R"(parent file map exists but does not contain the current DF "{}" in file "{}")", folderName.c_str(), mcurrentFile->GetName()));
    return nullptr;
  }

  if (mParentFile) {
    // Is this still the corresponding to the correct file?
    if (parentFileName->GetString().CompareTo(mParentFile->mcurrentFile->GetName()) == 0) {
      return mParentFile;
    } else {
      mParentFile->closeInputFile();
      delete mParentFile;
      mParentFile = nullptr;
    }
  }

  if (mLevel == mAllowedParentLevel) {
    throw std::runtime_error(fmt::format(R"(parent file requested but we are already at level {} of maximal allowed level {} for DF "{}" in file "{}")", mLevel, mAllowedParentLevel, folderName.c_str(), mcurrentFile->GetName()));
  }

  LOGP(info, "Opening parent file {} for DF {}", parentFileName->GetString().Data(), folderName.c_str());
  mParentFile = new DataInputDescriptor(mAlienSupport, mLevel + 1, mMonitoring, mAllowedParentLevel, mParentFileReplacement);
  mParentFile->mdefaultFilenamesPtr = new std::vector<FileNameHolder*>;
  mParentFile->mdefaultFilenamesPtr->emplace_back(makeFileNameHolder(parentFileName->GetString().Data()));
  mParentFile->fillInputfiles();
  mParentFile->setFile(0);
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

void DataInputDescriptor::printFileStatistics()
{
  int64_t wait_time = (int64_t)uv_hrtime() - (int64_t)mCurrentFileStartedAt - (int64_t)mIOTime;
  if (wait_time < 0) {
    wait_time = 0;
  }
  std::string monitoringInfo(fmt::format("lfn={},size={},total_df={},read_df={},read_bytes={},read_calls={},io_time={:.1f},wait_time={:.1f},level={}", mcurrentFile->GetName(),
                                         mcurrentFile->GetSize(), getTimeFramesInFile(mCurrentFileID), getReadTimeFramesInFile(mCurrentFileID), mcurrentFile->GetBytesRead(), mcurrentFile->GetReadCalls(),
                                         ((float)mIOTime / 1e9), ((float)wait_time / 1e9), mLevel));
#if __has_include(<TJAlienFile.h>)
  auto alienFile = dynamic_cast<TJAlienFile*>(mcurrentFile);
  if (alienFile) {
    monitoringInfo += fmt::format(",se={},open_time={:.1f}", alienFile->GetSE(), alienFile->GetElapsed());
  }
#endif
  mMonitoring->send(o2::monitoring::Metric{monitoringInfo, "aod-file-read-info"}.addTag(o2::monitoring::tags::Key::Subsystem, o2::monitoring::tags::Value::DPL));
  LOGP(info, "Read info: {}", monitoringInfo);
}

void DataInputDescriptor::closeInputFile()
{
  if (mcurrentFile) {
    if (mParentFile) {
      mParentFile->closeInputFile();
      delete mParentFile;
      mParentFile = nullptr;
    }

    delete mParentFileMap;
    mParentFileMap = nullptr;

    printFileStatistics();
    mcurrentFile->Close();
    delete mcurrentFile;
    mcurrentFile = nullptr;
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
  auto dfList = mfilenames[file]->listOfTimeFrameKeys;
  auto it = std::find(dfList.begin(), dfList.end(), dfName);
  if (it == dfList.end()) {
    return -1;
  }
  return it - dfList.begin();
}

bool DataInputDescriptor::readTree(DataAllocator& outputs, header::DataHeader dh, int counter, int numTF, std::string treename, size_t& totalSizeCompressed, size_t& totalSizeUncompressed)
{
  auto ioStart = uv_hrtime();

  auto fileAndFolder = getFileFolder(counter, numTF);
  if (!fileAndFolder.file) {
    return false;
  }

  auto fullpath = fileAndFolder.folderName + "/" + treename;
  auto tree = (TTree*)fileAndFolder.file->Get(fullpath.c_str());

  if (!tree) {
    LOGP(debug, "Could not find tree {}. Trying in parent file.", fullpath.c_str());
    auto parentFile = getParentFile(counter, numTF);
    if (parentFile != nullptr) {
      int parentNumTF = parentFile->findDFNumber(0, fileAndFolder.folderName);
      if (parentNumTF == -1) {
        throw std::runtime_error(fmt::format(R"(DF {} listed in parent file map but not found in the corresponding file "{}")", fileAndFolder.folderName, parentFile->mcurrentFile->GetName()));
      }
      // first argument is 0 as the parent file object contains only 1 file
      return parentFile->readTree(outputs, dh, 0, parentNumTF, treename, totalSizeCompressed, totalSizeUncompressed);
    }
    throw std::runtime_error(fmt::format(R"(Couldn't get TTree "{}" from "{}". Please check https://aliceo2group.github.io/analysis-framework/docs/troubleshooting/treenotfound.html for more information.)", fileAndFolder.folderName + "/" + treename, fileAndFolder.file->GetName()));
  }

  // create table output
  auto o = Output(dh);
  auto& t2t = outputs.make<TreeToTable>(o);

  // add branches to read
  // fill the table
  auto colnames = getColumnNames(dh);
  t2t.setLabel(tree->GetName());
  if (colnames.size() == 0) {
    totalSizeCompressed += tree->GetZipBytes();
    totalSizeUncompressed += tree->GetTotBytes();
    t2t.addAllColumns(tree);
  } else {
    for (auto& colname : colnames) {
      TBranch* branch = tree->GetBranch(colname.c_str());
      totalSizeCompressed += branch->GetZipBytes("*");
      totalSizeUncompressed += branch->GetTotBytes("*");
    }
    t2t.addAllColumns(tree, std::move(colnames));
  }
  t2t.fill(tree);
  delete tree;

  mIOTime += (uv_hrtime() - ioStart);

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

std::unique_ptr<TTreeReader> DataInputDirector::getTreeReader(header::DataHeader dh, int counter, int numTF, std::string treename)
{
  std::unique_ptr<TTreeReader> reader = nullptr;
  auto didesc = getDataInputDescriptor(dh);
  // if NOT match then use defaultDataInputDescriptor
  if (!didesc) {
    didesc = mdefaultDataInputDescriptor;
  }

  auto fileAndFolder = didesc->getFileFolder(counter, numTF);
  if (fileAndFolder.file) {
    treename = fileAndFolder.folderName + "/" + treename;
    reader = std::make_unique<TTreeReader>(treename.c_str(), fileAndFolder.file);
    if (!reader) {
      throw std::runtime_error(fmt::format(R"(Couldn't create TTreeReader for tree "{}" in file "{}")", treename, fileAndFolder.file->GetName()));
    }
  }

  return reader;
}

FileAndFolder DataInputDirector::getFileFolder(header::DataHeader dh, int counter, int numTF)
{
  auto didesc = getDataInputDescriptor(dh);
  // if NOT match then use defaultDataInputDescriptor
  if (!didesc) {
    didesc = mdefaultDataInputDescriptor;
  }

  return didesc->getFileFolder(counter, numTF);
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

  return didesc->getTimeFrameNumber(counter, numTF);
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

  return didesc->readTree(outputs, dh, counter, numTF, treename, totalSizeCompressed, totalSizeUncompressed);
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
