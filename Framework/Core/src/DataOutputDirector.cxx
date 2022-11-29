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
#include "Framework/DataOutputDirector.h"
#include "Framework/Logger.h"

#include <filesystem>

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"

#include "TMap.h"
#include "TObjString.h"

namespace fs = std::filesystem;

namespace o2
{
namespace framework
{
using namespace rapidjson;

DataOutputDescriptor::DataOutputDescriptor(std::string inString)
{
  // inString is an item consisting of 4 parts which are separated by a ':'
  // "origin/description/subSpec:treename:col1/col2/col3:filename"
  // the 1st part is used to create a DataDescriptorMatcher
  // the other parts are used to fill treename, colnames, and filename
  // remove all spaces
  auto cleanString = remove_ws(inString);

  // reset
  treename = "";
  colnames.clear();
  mfilenameBase = "";

  // analyze the  parts of the input string
  static const std::regex delim1(":");
  std::sregex_token_iterator end;
  std::sregex_token_iterator iter1(cleanString.begin(),
                                   cleanString.end(),
                                   delim1,
                                   -1);

  // create the DataDescriptorMatcher
  if (iter1 == end) {
    return;
  }
  auto tableString = iter1->str();
  matcher = DataDescriptorQueryBuilder::buildNode(tableString);

  // get the table name
  auto tableItems = DataDescriptorQueryBuilder::getTokens(tableString);
  if (!std::string(tableItems[2]).empty()) {
    tablename = tableItems[2];
  }
  if (!std::string(tableItems[3]).empty() && std::atoi(std::string(tableItems[3]).c_str()) > 0) {
    version = std::string{"_"}.append(std::string(3 - std::string(tableItems[3]).length(), '0')).append(std::string(tableItems[3]));
  }

  // get the tree name
  // default tree name is the O2 + table name (lower case)
  treename = tablename;
  std::transform(treename.begin(), treename.end(), treename.begin(), [](unsigned char c) { return std::tolower(c); });
  treename = std::string("O2") + treename + version;
  ++iter1;
  if (iter1 == end) {
    return;
  }
  if (!iter1->str().empty()) {
    treename = iter1->str();
  }

  // get column names
  ++iter1;
  if (iter1 == end) {
    return;
  }
  if (!iter1->str().empty()) {
    auto cns = iter1->str();

    static const std::regex delim2("/");
    std::sregex_token_iterator iter2(cns.begin(),
                                     cns.end(),
                                     delim2,
                                     -1);
    for (; iter2 != end; ++iter2) {
      if (!iter2->str().empty()) {
        colnames.emplace_back(iter2->str());
      }
    }
  }

  // get the file name base
  ++iter1;
  if (iter1 == end) {
    return;
  }
  if (!iter1->str().empty()) {
    mfilenameBase = iter1->str();
  }
}

std::string DataOutputDescriptor::getFilenameBase()
{
  return (mfilenameBase.empty() && mfilenameBasePtr) ? (std::string)*mfilenameBasePtr : mfilenameBase;
}

void DataOutputDescriptor::printOut()
{
  LOGP(info, "DataOutputDescriptor");
  LOGP(info, "  Table name     : {}", tablename);
  LOGP(info, "  File name base : {}", getFilenameBase());
  LOGP(info, "  Tree name      : {}", treename);
  if (colnames.empty()) {
    LOGP(info, "  Columns        : \"all\"");
  } else {
    LOGP(info, "  Columns        : {}", colnames.size());
  }
  for (auto cn : colnames) {
    LOGP(info, "    {}", cn);
  }
}

std::string DataOutputDescriptor::remove_ws(const std::string& s)
{
  std::string s_wns;
  for (auto c : s) {
    if (!std::isspace(c)) {
      s_wns += c;
    }
  }
  return s_wns;
}

DataOutputDirector::DataOutputDirector()
{
  mfilenameBase = std::string("");
}

void DataOutputDirector::reset()
{
  mDataOutputDescriptors.clear();
  mfilenameBases.clear();
  mtreeFilenames.clear();
  closeDataFiles();
  mfilePtrs.clear();
  mfilenameBase = std::string("");
  mfileCounter = 1;
};

void DataOutputDirector::readString(std::string const& keepString)
{
  // the keep-string keepString consists of ','-separated items
  // create for each item a corresponding DataOutputDescriptor
  static const std::regex delim(",");
  std::sregex_token_iterator end;
  std::sregex_token_iterator iter(keepString.begin(),
                                  keepString.end(),
                                  delim,
                                  -1);

  // loop over ','-separated items
  for (; iter != end; ++iter) {
    auto itemString = iter->str();

    // create a new DataOutputDescriptor and add it to the list
    auto dodesc = new DataOutputDescriptor(itemString);
    if (dodesc->getFilenameBase().empty()) {
      dodesc->setFilenameBase(mfilenameBasePtr);
    }
    mDataOutputDescriptors.emplace_back(dodesc);
    mfilenameBases.emplace_back(dodesc->getFilenameBase());
    mtreeFilenames.emplace_back(dodesc->treename + dodesc->getFilenameBase());
  }

  // the combination [tree name/file name] must be unique
  // throw exception if this is not the case
  auto it = std::unique(mtreeFilenames.begin(), mtreeFilenames.end());
  if (it != mtreeFilenames.end()) {
    printOut();
    LOGP(fatal, "Dublicate tree names in a file!");
  }

  // make unique/sorted list of filenameBases
  std::sort(mfilenameBases.begin(), mfilenameBases.end());
  auto last = std::unique(mfilenameBases.begin(), mfilenameBases.end());
  mfilenameBases.erase(last, mfilenameBases.end());

  // prepare list mfilePtrs of TFile
  for (auto fn : mfilenameBases) {
    mfilePtrs.emplace_back(new TFile());
    mParentMaps.emplace_back(new TMap());
  }
}

// creates a keep string from a InputSpec
std::string SpectoString(InputSpec input)
{
  std::string keepString;
  std::string delim("/");

  auto matcher = DataSpecUtils::asConcreteDataMatcher(input);
  keepString += matcher.origin.str + delim;
  keepString += matcher.description.str + delim;
  keepString += std::to_string(matcher.subSpec);

  return keepString;
}

void DataOutputDirector::readSpecs(std::vector<InputSpec> inputs)
{
  for (auto input : inputs) {
    auto keepString = SpectoString(input);
    readString(keepString);
  }
}

std::tuple<std::string, std::string, std::string, float, int> DataOutputDirector::readJson(std::string const& fnjson)
{
  // open the file
  FILE* fjson = fopen(fnjson.c_str(), "r");
  if (!fjson) {
    LOGP(info, "Could not open JSON file \"{}\"", fnjson);
    return memptyanswer;
  }

  // create streamer
  char readBuffer[65536];
  FileReadStream jsonStream(fjson, readBuffer, sizeof(readBuffer));

  // parse the json file
  Document jsonDocument;
  jsonDocument.ParseStream(jsonStream);
  auto [rdn, dfn, fmode, mfs, ntfm] = readJsonDocument(&jsonDocument);

  // clean up
  fclose(fjson);

  return std::make_tuple(rdn, dfn, fmode, mfs, ntfm);
}

std::tuple<std::string, std::string, std::string, float, int> DataOutputDirector::readJsonString(std::string const& jsonString)
{
  // parse the json string
  Document jsonDocument;
  jsonDocument.Parse(jsonString.c_str());
  auto [rdn, dfn, fmode, mfs, ntfm] = readJsonDocument(&jsonDocument);

  return std::make_tuple(rdn, dfn, fmode, mfs, ntfm);
}

std::tuple<std::string, std::string, std::string, float, int> DataOutputDirector::readJsonDocument(Document* jsonDocument)
{
  std::string smc(":");
  std::string slh("/");
  const char* itemName;

  // initialisations
  std::string resdir(".");
  std::string dfn("");
  std::string fmode("");
  float maxfs = -1.;
  int ntfm = -1;

  // is it a proper json document?
  if (jsonDocument->HasParseError()) {
    LOGP(error, "Check the JSON document! There is a problem with the format!");
    return memptyanswer;
  }

  // OutputDirector
  itemName = "OutputDirector";
  const Value& dodirItem = (*jsonDocument)[itemName];
  if (!dodirItem.IsObject()) {
    LOGP(info, "No \"{}\" object found in the JSON document!", itemName);
    return memptyanswer;
  }

  // now read various items
  itemName = "debugmode";
  if (dodirItem.HasMember(itemName)) {
    if (dodirItem[itemName].IsBool()) {
      mdebugmode = dodirItem[itemName].GetBool();
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a boolean!", itemName);
      return memptyanswer;
    }
  } else {
    mdebugmode = false;
  }

  if (mdebugmode) {
    StringBuffer buffer;
    buffer.Clear();
    Writer<rapidjson::StringBuffer> writer(buffer);
    dodirItem.Accept(writer);
  }

  itemName = "resdir";
  if (dodirItem.HasMember(itemName)) {
    if (dodirItem[itemName].IsString()) {
      resdir = dodirItem[itemName].GetString();
      setResultDir(resdir);
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a string!", itemName);
      return memptyanswer;
    }
  }

  itemName = "resfile";
  if (dodirItem.HasMember(itemName)) {
    if (dodirItem[itemName].IsString()) {
      dfn = dodirItem[itemName].GetString();
      setFilenameBase(dfn);
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a string!", itemName);
      return memptyanswer;
    }
  }

  itemName = "resfilemode";
  if (dodirItem.HasMember(itemName)) {
    if (dodirItem[itemName].IsString()) {
      fmode = dodirItem[itemName].GetString();
      setFileMode(fmode);
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a string!", itemName);
      return memptyanswer;
    }
  }

  itemName = "maxfilesize";
  if (dodirItem.HasMember(itemName)) {
    if (dodirItem[itemName].IsNumber()) {
      maxfs = dodirItem[itemName].GetFloat();
      setMaximumFileSize(maxfs);
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a number!", itemName);
      return memptyanswer;
    }
  }

  itemName = "ntfmerge";
  if (dodirItem.HasMember(itemName)) {
    if (dodirItem[itemName].IsNumber()) {
      ntfm = dodirItem[itemName].GetInt();
      setNumberTimeFramesToMerge(ntfm);
    } else {
      LOGP(error, "Check the JSON document! Item \"{}\" must be a number!", itemName);
      return memptyanswer;
    }
  }

  itemName = "OutputDescriptors";
  if (dodirItem.HasMember(itemName)) {
    if (!dodirItem[itemName].IsArray()) {
      LOGP(error, "Check the JSON document! Item \"{}\" must be an array!", itemName);
      return memptyanswer;
    }

    // loop over DataOutputDescriptors
    for (auto& dodescItem : dodirItem[itemName].GetArray()) {
      if (!dodescItem.IsObject()) {
        LOGP(error, "Check the JSON document! \"{}\" must be objects!", itemName);
        return memptyanswer;
      }

      std::string dodString = "";
      itemName = "table";
      if (dodescItem.HasMember(itemName)) {
        if (dodescItem[itemName].IsString()) {
          dodString += dodescItem[itemName].GetString();
        } else {
          LOGP(error, "Check the JSON document! \"{}\" must be a string!", itemName);
          return memptyanswer;
        }
      }
      dodString += smc;
      itemName = "treename";
      if (dodescItem.HasMember(itemName)) {
        if (dodescItem[itemName].IsString()) {
          dodString += dodescItem[itemName].GetString();
        } else {
          LOGP(error, "Check the JSON document! \"{}\" must be a string!", itemName);
          return memptyanswer;
        }
      }
      dodString += smc;
      itemName = "columns";
      if (dodescItem.HasMember(itemName)) {
        if (dodescItem[itemName].IsArray()) {
          auto columnNames = dodescItem[itemName].GetArray();
          for (auto& c : columnNames) {
            dodString += (c == columnNames[0]) ? c.GetString() : slh + c.GetString();
          }
        } else {
          LOGP(error, "Check the JSON document! \"{}\" must be an array!", itemName);
          return memptyanswer;
        }
      }
      dodString += smc;
      itemName = "filename";
      if (dodescItem.HasMember(itemName)) {
        if (dodescItem[itemName].IsString()) {
          dodString += dodescItem[itemName].GetString();
        } else {
          LOGP(error, "Check the JSON document! \"{}\" must be a string!", itemName);
          return memptyanswer;
        }
      }

      // convert s to DataOutputDescription object
      readString(dodString);
    }
  }

  // print the DataOutputDirector
  if (mdebugmode) {
    printOut();
  }

  return std::make_tuple(resdir, dfn, fmode, maxfs, ntfm);
}

std::vector<DataOutputDescriptor*> DataOutputDirector::getDataOutputDescriptors(header::DataHeader dh)
{
  std::vector<DataOutputDescriptor*> result;

  // compute list of matching outputs
  data_matcher::VariableContext context;

  for (auto dodescr : mDataOutputDescriptors) {
    if (dodescr->matcher->match(dh, context)) {
      result.emplace_back(dodescr);
    }
  }

  return result;
}

std::vector<DataOutputDescriptor*> DataOutputDirector::getDataOutputDescriptors(InputSpec spec)
{
  std::vector<DataOutputDescriptor*> result;

  // compute list of matching outputs
  data_matcher::VariableContext context;
  auto concrete = std::get<ConcreteDataMatcher>(spec.matcher);

  for (auto dodescr : mDataOutputDescriptors) {
    if (dodescr->matcher->match(concrete, context)) {
      result.emplace_back(dodescr);
    }
  }

  return result;
}

FileAndFolder DataOutputDirector::getFileFolder(DataOutputDescriptor* dodesc, uint64_t folderNumber, std::string parentFileName)
{
  // initialisation
  FileAndFolder fileAndFolder;

  // search dodesc->filename in mfilenameBases and return corresponding filePtr
  auto it = std::find(mfilenameBases.begin(), mfilenameBases.end(), dodesc->getFilenameBase());
  if (it != mfilenameBases.end()) {
    int ind = std::distance(mfilenameBases.begin(), it);

    // open new output file
    if (!mfilePtrs[ind]->IsOpen()) {
      // output directory
      auto resdirname = mresultDirectory;
      // is the maximum-file-size check enabled?
      if (mmaxfilesize > 0.) {
        // subdirectory ./xxx
        char chcnt[4];
        std::snprintf(chcnt, sizeof(chcnt), "%03d", mfileCounter);
        resdirname += "/" + std::string(chcnt);
      }
      auto resdir = fs::path{resdirname.c_str()};

      if (!fs::is_directory(resdir)) {
        if (!fs::create_directories(resdir)) {
          LOGF(fatal, "Could not create output directory %s", resdirname.c_str());
        }
      }

      // complete file name
      auto fn = resdirname + "/" + mfilenameBases[ind] + ".root";
      delete mfilePtrs[ind];
      mParentMaps[ind]->Clear();
      mfilePtrs[ind] = TFile::Open(fn.c_str(), mfileMode.c_str(), "", 501);
    }
    fileAndFolder.file = mfilePtrs[ind];

    // check if folder DF_* exists
    fileAndFolder.folderName = "DF_" + std::to_string(folderNumber);
    auto key = fileAndFolder.file->GetKey(fileAndFolder.folderName.c_str());
    if (!key) {
      fileAndFolder.file->mkdir(fileAndFolder.folderName.c_str());
      // TODO not clear why we get a " " in case we sent empty over DPL, put the limit to 1 for now
      if (parentFileName.length() > 1) {
        mParentMaps[ind]->Add(new TObjString(fileAndFolder.folderName.c_str()), new TObjString(parentFileName.c_str()));
      }
    }
    fileAndFolder.file->cd(fileAndFolder.folderName.c_str());
  }

  return fileAndFolder;
}

bool DataOutputDirector::checkFileSizes()
{
  // is the maximum-file-size check enabled?
  if (mmaxfilesize <= 0.) {
    return true;
  }

  // current result directory
  char chcnt[4];
  std::snprintf(chcnt, sizeof(chcnt), "%03d", mfileCounter);
  std::string strcnt{chcnt};
  auto resdirname = mresultDirectory + "/" + strcnt;

  // loop over all files
  // if one file is large, then all files need to be closed
  for (int i = 0; i < mfilenameBases.size(); i++) {
    if (!mfilePtrs[i]) {
      continue;
    }
    // size of fn
    auto fn = resdirname + "/" + mfilenameBases[i] + ".root";
    auto resfile = fs::path{fn.c_str()};
    if (!fs::exists(resfile)) {
      continue;
    }
    auto fsize = (float)fs::file_size(resfile) / 1.E6; // MBytes
    LOGF(debug, "File %s: %f MBytes", fn.c_str(), fsize);
    if (fsize >= mmaxfilesize) {
      closeDataFiles();
      // increment the subdirectory counter
      mfileCounter++;
      return false;
    }
  }

  return true;
}

void DataOutputDirector::closeDataFiles()
{
  for (int i = 0; i < mfilePtrs.size(); i++) {
    auto filePtr = mfilePtrs[i];
    if (filePtr) {
      if (filePtr->IsOpen() && mParentMaps[i]->GetEntries() > 0) {
        filePtr->cd("/");
        filePtr->WriteObject(mParentMaps[i], "parentFiles");
      }
      filePtr->Close();
    }
  }
}

void DataOutputDirector::printOut()
{
  LOGP(info, "DataOutputDirector");
  LOGP(info, "  Output directory     : {}", mresultDirectory);
  LOGP(info, "  Default file name    : {}", mfilenameBase);
  LOGP(info, "  Maximum file size    : {} megabytes", mmaxfilesize);
  LOGP(info, "  Number of files      : {}", mfilenameBases.size());

  LOGP(info, "  DataOutputDescriptors: {}", mDataOutputDescriptors.size());
  for (auto const& ds : mDataOutputDescriptors) {
    ds->printOut();
  }

  LOGP(info, "  File name bases      :");
  for (auto const& fb : mfilenameBases) {
    LOGP(info, fb);
  }
}

void DataOutputDirector::setResultDir(std::string resDir)
{
  mresultDirectory = resDir;
}

void DataOutputDirector::setFilenameBase(std::string dfn)
{
  // reset
  mfilenameBase = dfn;

  mfilenameBases.clear();
  mtreeFilenames.clear();
  closeDataFiles();
  mfilePtrs.clear();

  // loop over DataOutputDescritors
  for (auto dodesc : mDataOutputDescriptors) {
    mfilenameBases.emplace_back(dodesc->getFilenameBase());
    mtreeFilenames.emplace_back(dodesc->treename + dodesc->getFilenameBase());
  }

  // the combination [tree name/file name] must be unique
  // throw exception if this is not the case
  auto it = std::unique(mtreeFilenames.begin(), mtreeFilenames.end());
  if (it != mtreeFilenames.end()) {
    printOut();
    LOG(fatal) << "Duplicate tree names in a file!";
  }

  // make unique/sorted list of filenameBases
  std::sort(mfilenameBases.begin(), mfilenameBases.end());
  auto last = std::unique(mfilenameBases.begin(), mfilenameBases.end());
  mfilenameBases.erase(last, mfilenameBases.end());

  // prepare list mfilePtrs of TFile
  for (auto fn : mfilenameBases) {
    mfilePtrs.emplace_back(new TFile());
    mParentMaps.emplace_back(new TMap());
  }
}

void DataOutputDirector::setMaximumFileSize(float maxfs)
{
  mmaxfilesize = maxfs;
}

} // namespace framework
} // namespace o2
