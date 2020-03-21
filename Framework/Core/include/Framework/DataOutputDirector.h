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

#include <regex>

namespace o2
{
namespace framework
{
namespace data_matcher
{

struct DataOutputDescriptor {
  /// Holds information concerning the writing of aod tables.
  /// The information includes the table specification, treename,
  /// columns to save, and the file name

  std::string tablename = "";
  std::string treename = "";
  std::string filename = "";
  std::vector<std::string> colnames;
  std::unique_ptr<DataDescriptorMatcher> matcher;

  DataOutputDescriptor(std::string sin)
  {
    // sin is an item consisting of 4 parts which are separated by a ':'
    // "origin/description/subSpec:treename:col1/col2/col3:filename"
    // the 1st part is used to create a DataDescriptorMatcher
    // the other parts are used to fill treename, colnames, and filename

    // remove all spaces
    auto s = remove_ws(sin);

    // reset
    treename = "";
    colnames.clear();
    filename = "";

    // analyze the  parts of the input string
    static const std::regex delim1(":");
    std::sregex_token_iterator end;
    std::sregex_token_iterator iter1(s.begin(),
                                     s.end(),
                                     delim1,
                                     -1);

    // create the DataDescriptorMatcher
    if (iter1 == end)
      return;
    auto a = iter1->str();
    matcher = DataDescriptorQueryBuilder::buildNode(a);

    // get the table name
    auto m = DataDescriptorQueryBuilder::getTokens(a);
    if (!std::string(m[2]).empty())
      tablename = m[2];

    // get the tree name
    // defaul tree name is the table name
    treename = tablename;
    ++iter1;
    if (iter1 == end)
      return;
    if (!iter1->str().empty())
      treename = iter1->str();

    // get column names
    ++iter1;
    if (iter1 == end)
      return;
    if (!iter1->str().empty()) {
      auto cns = iter1->str();

      static const std::regex delim2("/");
      std::sregex_token_iterator iter2(cns.begin(),
                                       cns.end(),
                                       delim2,
                                       -1);
      for (; iter2 != end; ++iter2)
        if (!iter2->str().empty())
          colnames.emplace_back(iter2->str());
    }

    // get the filename
    ++iter1;
    if (iter1 == end)
      return;
    if (!iter1->str().empty())
      filename = iter1->str();
  }

  void setFilename(std::string fn) { filename = fn; }

  void printOut()
  {
    LOG(INFO) << "DataOutputDescriptor";
    LOG(INFO) << "  table name: " << tablename.c_str();
    LOG(INFO) << "  file name : " << filename.c_str();
    LOG(INFO) << "  tree name : " << treename.c_str();
    LOG(INFO) << "  columns   : " << colnames.size();
    for (auto cn : colnames)
      LOG(INFO) << "  " << cn.c_str();
  }

  std::string remove_ws(const std::string& s)
  {
    std::string s_wns;
    for (auto c : s)
      if (!std::isspace(c))
        s_wns += c;
    return s_wns;
  }
};

struct DataOutputDirector {

  int ndod = 0;
  std::string defaultfname;
  std::vector<DataOutputDescriptor*> dodescrs;

  std::vector<std::string> tnfns;

  std::vector<std::string> fnames;
  std::vector<int> fcnts;
  std::vector<TFile*> fouts;

  DataOutputDirector();
  void reset();

  // fill the DataOutputDirector with information from a
  // keep-string
  void readString(std::string const& keepString);

  // fill the DataOutputDirector with information from a
  // list of InputSpec
  void readSpecs(std::vector<InputSpec> inputs);

  // fill the DataOutputDirector with information from a json file
  //readJson (std::string const& fnjson) {};

  // get matching DataOutputDescriptors
  std::vector<DataOutputDescriptor*> getDataOutputDescriptors(header::DataHeader dh);
  std::vector<DataOutputDescriptor*> getDataOutputDescriptors(InputSpec spec);

  // get the matching TFile
  TFile* getDataOutputFile(DataOutputDescriptor* dod,
                           int ntf, int ntfmerge, std::string filemode);
  void closeDataOutputFiles();

  void setDefaultfname(std::string dfn) { defaultfname = dfn; }

  void printOut();
};

} // namespace data_matcher
} // namespace framework
} // namespace o2

#endif // o2_framework_DataOutputDirector_H_INCLUDED
