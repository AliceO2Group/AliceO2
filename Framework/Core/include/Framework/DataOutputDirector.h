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

  void setFilename(std::string fn) { filename = fn; }
  void setFilename(std::string* fnptr) { dfnptr = fnptr; }
  std::string getFilename();

  void printOut();

 private:
  std::string filename;
  std::string* dfnptr = nullptr;

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
  TFile* getDataOutputFile(DataOutputDescriptor* dod,
                           int ntf, int ntfmerge, std::string filemode);
  void closeDataOutputFiles();

  void setDefaultfname(std::string dfn);

  void printOut();

 private:
  int ndod = 0;
  std::string defaultfname;
  std::string* const dfnptr = &defaultfname;
  std::vector<DataOutputDescriptor*> dodescrs;
  std::vector<std::string> tnfns;
  std::vector<std::string> fnames;
  std::vector<int> fcnts;
  std::vector<TFile*> fouts;

  std::tuple<std::string, std::string, int> readJsonDocument(Document* doc);
};

} // namespace framework
} // namespace o2

#endif // o2_framework_DataOutputDirector_H_INCLUDED
