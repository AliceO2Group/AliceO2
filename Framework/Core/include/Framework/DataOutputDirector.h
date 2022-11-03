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
#ifndef o2_framework_DataOutputDirector_H_INCLUDED
#define o2_framework_DataOutputDirector_H_INCLUDED

#include "TFile.h"

#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InputSpec.h"

#include "rapidjson/fwd.h"

class TFile;

namespace o2::framework
{
using namespace rapidjson;

struct FileAndFolder {
  TFile* file = nullptr;
  std::string folderName = "";
};

struct DataOutputDescriptor {
  /// Holds information concerning the writing of aod tables.
  /// The information includes the table specification, treename,
  /// columns to save, and the file name

  std::string tablename;
  std::string treename;
  std::string version;
  std::vector<std::string> colnames;
  std::unique_ptr<data_matcher::DataDescriptorMatcher> matcher;

  DataOutputDescriptor(std::string sin);

  void setFilenameBase(std::string fn) { mfilenameBase = fn; }
  void setFilenameBase(std::string* fnptr) { mfilenameBasePtr = fnptr; }
  std::string getFilenameBase();

  void printOut();

 private:
  std::string mfilenameBase;
  std::string* mfilenameBasePtr = nullptr;

  std::string remove_ws(const std::string& s);
};

struct DataOutputDirector {
  /// Holds a list of DataOutputDescriptor and a list of output files
  /// Provides functionality to access the matching DataOutputDescriptor
  /// and the related output file

  DataOutputDirector();
  void reset();

  // fill the DataOutputDirector with information from a
  // keep-string
  void readString(std::string const& keepString);

  // fill the DataOutputDirector with information from a
  // list of InputSpec
  void readSpecs(std::vector<InputSpec> inputs);

  // fill the DataOutputDirector with information from a json file
  std::tuple<std::string, std::string, int> readJson(std::string const& fnjson);
  std::tuple<std::string, std::string, int> readJsonString(std::string const& stjson);

  // read/write private members
  int getNumberTimeFramesToMerge() { return mnumberTimeFramesToMerge; }
  void setNumberTimeFramesToMerge(int ntfmerge) { mnumberTimeFramesToMerge = ntfmerge > 0 ? ntfmerge : 1; }
  std::string getFileMode() { return mfileMode; }
  void setFileMode(std::string filemode) { mfileMode = filemode; }

  // get matching DataOutputDescriptors
  std::vector<DataOutputDescriptor*> getDataOutputDescriptors(header::DataHeader dh);
  std::vector<DataOutputDescriptor*> getDataOutputDescriptors(InputSpec spec);

  // get the matching TFile
  FileAndFolder getFileFolder(DataOutputDescriptor* dodesc, uint64_t folderNumber, std::string parentFileName);

  void closeDataFiles();

  void setFilenameBase(std::string dfn);

  void printOut();

 private:
  std::string mfilenameBase;
  std::string* const mfilenameBasePtr = &mfilenameBase;
  std::vector<DataOutputDescriptor*> mDataOutputDescriptors;
  std::vector<std::string> mtreeFilenames;
  std::vector<std::string> mfilenameBases;
  std::vector<TFile*> mfilePtrs;
  std::vector<TMap*> mParentMaps;
  bool mdebugmode = false;
  int mnumberTimeFramesToMerge = 1;
  std::string mfileMode = "RECREATE";

  std::tuple<std::string, std::string, int> readJsonDocument(Document* doc);
  const std::tuple<std::string, std::string, int> memptyanswer = std::make_tuple(std::string(""), std::string(""), -1);
};

} // namespace o2::framework

#endif // o2_framework_DataOutputDirector_H_INCLUDED
