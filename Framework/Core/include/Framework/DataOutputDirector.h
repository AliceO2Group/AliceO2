// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace framework
{
using namespace rapidjson;

struct DataOutputDescriptor {
  /// Holds information concerning the writing of aod tables.
  /// The information includes the table specification, treename,
  /// columns to save, and the file name

  std::string tablename = "";
  std::string treename = "";
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

  // get matching DataOutputDescriptors
  std::vector<DataOutputDescriptor*> getDataOutputDescriptors(header::DataHeader dh);
  std::vector<DataOutputDescriptor*> getDataOutputDescriptors(InputSpec spec);

  // get the matching TFile
  std::tuple<TFile*, std::string> getFileFolder(DataOutputDescriptor* dodesc,
                                                int ntf, int ntfmerge,
                                                std::string filemode);

  void closeDataFiles();

  void setFilenameBase(std::string dfn);

  void printOut();

 private:
  std::string mfilenameBase;
  std::string* const mfilenameBasePtr = &mfilenameBase;
  std::vector<DataOutputDescriptor*> mDataOutputDescriptors;
  std::vector<std::string> mtreeFilenames;
  std::vector<std::string> mfilenameBases;
  std::vector<int> mfolderCounts;
  std::vector<TFile*> mfilePtrs;
  bool mdebugmode = false;

  std::tuple<std::string, std::string, int> readJsonDocument(Document* doc);
  const std::tuple<std::string, std::string, int> memptyanswer = std::make_tuple(std::string(""), std::string(""), -1);
};

} // namespace framework
} // namespace o2

#endif // o2_framework_DataOutputDirector_H_INCLUDED
