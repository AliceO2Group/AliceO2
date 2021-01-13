// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataInputDirector.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/Logger.h"
#include "AnalysisDataModelHelpers.h"

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"

#include "TGrid.h"
#include "TObjString.h"

namespace o2
{
namespace framework
{
using namespace rapidjson;

FileNameHolder* makeFileNameHolder(std::string fileName)
{
  auto fileNameHolder = new FileNameHolder();
  fileNameHolder->fileName = fileName;

  return fileNameHolder;
}

DataInputDescriptor::DataInputDescriptor(bool alienSupport)
{
  mAlienSupport = alienSupport;
}

void DataInputDescriptor::printOut()
{
  LOGP(INFO, "DataInputDescriptor");
  LOGP(INFO, "  Table name        : {}", tablename);
  LOGP(INFO, "  Tree name         : {}", treename);
  LOGP(INFO, "  Input files file  : {}", getInputfilesFilename());
  LOGP(INFO, "  File name regex   : {}", getFilenamesRegexString());
  LOGP(INFO, "  Input files       : {}", mfilenames.size());
  for (auto fn : mfilenames) {
    LOGP(INFO, "    {} {}", fn->fileName, fn->numberOfTimeFrames);
  }
  LOGP(INFO, "  Total number of TF: {}", getNumberTimeFrames());
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
    LOGP(DEBUG, "AliEn file requested. Enabling support.");
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
    if (mcurrentFile->GetName() != filename) {
      closeInputFile();
      mcurrentFile = TFile::Open(filename.c_str());
    }
  } else {
    mcurrentFile = TFile::Open(filename.c_str());
  }
  if (!mcurrentFile) {
    throw std::runtime_error(fmt::format("Couldn't open file \"{}\"!", filename));
  }
  mcurrentFile->SetReadaheadSize(50 * 1024 * 1024);

  // get the directory names
  if (mfilenames[counter]->numberOfTimeFrames <= 0) {
    std::regex TFRegex = std::regex("TF_[0-9]+");
    TList* keyList = mcurrentFile->GetListOfKeys();

    // extract TF numbers and sort accordingly
    for (auto key : *keyList) {
      if (std::regex_match(((TObjString*)key)->GetString().Data(), TFRegex)) {
        auto folderNumber = std::stoul(std::string(((TObjString*)key)->GetString().Data()).substr(3));
        mfilenames[counter]->listOfTimeFrameNumbers.emplace_back(folderNumber);
      }
    }
    std::sort(mfilenames[counter]->listOfTimeFrameNumbers.begin(), mfilenames[counter]->listOfTimeFrameNumbers.end());

    for (auto folderNumber : mfilenames[counter]->listOfTimeFrameNumbers) {
      auto folderName = "TF_" + std::to_string(folderNumber);
      mfilenames[counter]->listOfTimeFrameKeys.emplace_back(folderName);
    }
    mfilenames[counter]->numberOfTimeFrames = mfilenames[counter]->listOfTimeFrameKeys.size();
  }

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

  return fileAndFolder;
}

int DataInputDescriptor::getTimeFramesInFile(int counter)
{
  return mfilenames.at(counter)->numberOfTimeFrames;
}

void DataInputDescriptor::closeInputFile()
{
  if (mcurrentFile) {
    mcurrentFile->Close();
    mcurrentFile = nullptr;
    delete mcurrentFile;
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
      while (std::getline(filelist, fileName)) {
        // remove white spaces, empty lines are skipped
        fileName.erase(std::remove_if(fileName.begin(), fileName.end(), ::isspace), fileName.end());
        if (!fileName.empty() && (getFilenamesRegexString().empty() ||
                                  std::regex_match(fileName, getFilenamesRegex()))) {
          addFileNameHolder(makeFileNameHolder(fileName));
        }
      }
    } catch (...) {
      LOGP(ERROR, "Check the input files file! Unable to process \"{}\"!", getInputfilesFilename());
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

DataInputDirector::DataInputDirector()
{
  createDefaultDataInputDescriptor();
}

DataInputDirector::DataInputDirector(std::string inputFile)
{
  if (inputFile.size() && inputFile[0] == '@') {
    inputFile.erase(0, 1);
    setInputfilesFile(inputFile);
  } else {
    mdefaultInputFiles.emplace_back(makeFileNameHolder(inputFile));
  }

  createDefaultDataInputDescriptor();
}

DataInputDirector::DataInputDirector(std::vector<std::string> inputFiles)
{
  for (auto inputFile : inputFiles) {
    mdefaultInputFiles.emplace_back(makeFileNameHolder(inputFile));
  }

  createDefaultDataInputDescriptor();
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
  mdefaultDataInputDescriptor = new DataInputDescriptor(mAlienSupport);

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
    LOGP(ERROR, "Could not open JSON file \"{}\"!", fnjson);
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
  int ntfm = -1;
  const char* itemName;

  // is it a proper json document?
  if (jsonDoc->HasParseError()) {
    LOGP(ERROR, "Check the JSON document! There is a problem with the format!");
    return false;
  }

  // InputDirector
  itemName = "InputDirector";
  const Value& didirItem = (*jsonDoc)[itemName];
  if (!didirItem.IsObject()) {
    LOGP(INFO, "No \"{}\" object found in the JSON document!", itemName);
    return true;
  }

  // now read various items
  itemName = "debugmode";
  if (didirItem.HasMember(itemName)) {
    if (didirItem[itemName].IsBool()) {
      mDebugMode = (didirItem[itemName].GetBool());
    } else {
      LOGP(ERROR, "Check the JSON document! Item \"{}\" must be a boolean!", itemName);
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
    LOGP(INFO, "InputDirector object: {}", std::string(buffer.GetString()));
  }

  itemName = "fileregex";
  if (didirItem.HasMember(itemName)) {
    if (didirItem[itemName].IsString()) {
      setFilenamesRegex(didirItem[itemName].GetString());
    } else {
      LOGP(ERROR, "Check the JSON document! Item \"{}\" must be a string!", itemName);
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
      LOGP(ERROR, "Check the JSON document! Item \"{}\" must be a string or an array!", itemName);
      return false;
    }
  }

  itemName = "InputDescriptors";
  if (didirItem.HasMember(itemName)) {
    if (!didirItem[itemName].IsArray()) {
      LOGP(ERROR, "Check the JSON document! Item \"{}\" must be an array!", itemName);
      return false;
    }

    // loop over DataInputDescriptors
    for (auto& didescItem : didirItem[itemName].GetArray()) {
      if (!didescItem.IsObject()) {
        LOGP(ERROR, "Check the JSON document! \"{}\" must be objects!", itemName);
        return false;
      }
      // create a new dataInputDescriptor
      auto didesc = new DataInputDescriptor(mAlienSupport);
      didesc->setDefaultInputfiles(&mdefaultInputFiles);

      itemName = "table";
      if (didescItem.HasMember(itemName)) {
        if (didescItem[itemName].IsString()) {
          didesc->tablename = didescItem[itemName].GetString();
          didesc->matcher = DataDescriptorQueryBuilder::buildNode(didesc->tablename);
        } else {
          LOGP(ERROR, "Check the JSON document! Item \"{}\" must be a string!", itemName);
          return false;
        }
      } else {
        LOGP(ERROR, "Check the JSON document! Item \"{}\" is missing!", itemName);
        return false;
      }

      itemName = "treename";
      if (didescItem.HasMember(itemName)) {
        if (didescItem[itemName].IsString()) {
          didesc->treename = didescItem[itemName].GetString();
        } else {
          LOGP(ERROR, "Check the JSON document! Item \"{}\" must be a string!", itemName);
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
          LOGP(ERROR, "Check the JSON document! Item \"{}\" must be a string!", itemName);
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
          LOGP(ERROR, "Check the JSON document! Item \"{}\" must be a string or an array!", itemName);
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
        LOGP(INFO, "This DataInputDescriptor is ignored because its file list is empty!");
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

TTree* DataInputDirector::getDataTree(header::DataHeader dh, int counter, int numTF)
{
  std::string treename;
  TTree* tree = nullptr;

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

  auto fileAndFolder = didesc->getFileFolder(counter, numTF);
  if (fileAndFolder.file) {
    treename = fileAndFolder.folderName + "/" + treename;
    tree = (TTree*)fileAndFolder.file->Get(treename.c_str());
    if (!tree) {
      throw std::runtime_error(fmt::format(R"(Couldn't get TTree "{}" from "{}")", treename, fileAndFolder.file->GetName()));
    }
  }

  return tree;
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
  LOGP(INFO, "DataInputDirector");
  LOGP(INFO, "  Default input files file   : {}", minputfilesFile);
  LOGP(INFO, "  Default file name regex    : {}", mFilenameRegex);
  LOGP(INFO, "  Default file names         : {}", mdefaultInputFiles.size());
  for (auto const& fn : mdefaultInputFiles) {
    LOGP(INFO, "    {} {}", fn->fileName, fn->numberOfTimeFrames);
  }
  LOGP(INFO, "  Default DataInputDescriptor:");
  mdefaultDataInputDescriptor->printOut();
  LOGP(INFO, "  DataInputDescriptors       : {}", getNumberInputDescriptors());
  for (auto const& didesc : mdataInputDescriptors) {
    didesc->printOut();
  }
}

} // namespace framework
} // namespace o2
