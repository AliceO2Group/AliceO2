// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataOutputDirector.h"

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

namespace o2
{
namespace framework
{
using namespace rapidjson;

DataOutputDescriptor::DataOutputDescriptor(std::string sin)
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
  if (iter1 == end) {
    return;
  }
  auto a = iter1->str();
  matcher = DataDescriptorQueryBuilder::buildNode(a);

  // get the table name
  auto m = DataDescriptorQueryBuilder::getTokens(a);
  if (!std::string(m[2]).empty()) {
    tablename = m[2];
  }

  // get the tree name
  // defaul tree name is the table name
  treename = tablename;
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

  // get the filename
  ++iter1;
  if (iter1 == end) {
    return;
  }
  if (!iter1->str().empty()) {
    filename = iter1->str();
  }
}

std::string DataOutputDescriptor::getFilename()
{
  return (filename.empty() && dfnptr) ? (std::string)*dfnptr : filename;
}

void DataOutputDescriptor::printOut()
{
  LOG(INFO) << "DataOutputDescriptor";
  LOG(INFO) << "  table name: " << tablename;
  LOG(INFO) << "  file name : " << getFilename();
  LOG(INFO) << "  tree name : " << treename;
  if (colnames.empty()) {
    LOG(INFO) << "  columns   : all";
  } else {
    LOG(INFO) << "  columns   : " << colnames.size();
  }
  for (auto cn : colnames)
    LOG(INFO) << "  " << cn;
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
  defaultfname = std::string("");
}

void DataOutputDirector::reset()
{
  dodescrs.clear();
  fnames.clear();
  tnfns.clear();
  closeDataOutputFiles();
  fouts.clear();
  fcnts.clear();
  defaultfname = std::string("");
  ndod = 0;
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
    auto s = iter->str();

    // create a new DataOutputDescriptor and add it to the list
    auto dod = new DataOutputDescriptor(s);
    if (dod->getFilename().empty()) {
      dod->setFilename(dfnptr);
    }
    dodescrs.emplace_back(dod);
    fnames.emplace_back(dod->getFilename());
    tnfns.emplace_back(dod->treename + dod->getFilename());
  }

  // the combination [tree name/file name] must be unique
  // throw exception if this is not the case
  auto it = std::unique(tnfns.begin(), tnfns.end());
  if (it != tnfns.end()) {
    printOut();
    LOG(FATAL) << "Dublicate tree names in a file!";
  }

  // make unique/sorted list of fnames
  std::sort(fnames.begin(), fnames.end());
  auto last = std::unique(fnames.begin(), fnames.end());
  fnames.erase(last, fnames.end());

  // prepare list fouts of TFile and fcnts
  for (auto fn : fnames) {
    fouts.emplace_back(new TFile());
    fcnts.emplace_back(-1);
  }

  // number of DataOutputDescriptors
  ndod = dodescrs.size();
}

// creates a keep string from a InputSpec
std::string SpectoString(InputSpec input)
{
  std::string s;
  std::string delim("/");

  auto matcher = DataSpecUtils::asConcreteDataMatcher(input);
  s += matcher.origin.str + delim;
  s += matcher.description.str + delim;
  s += std::to_string(matcher.subSpec);

  return s;
}

void DataOutputDirector::readSpecs(std::vector<InputSpec> inputs)
{
  for (auto input : inputs) {
    auto s = SpectoString(input);
    readString(s);
  }
}

std::tuple<std::string, std::string, int> DataOutputDirector::readJson(std::string const& fnjson)
{
  // open the file
  FILE* f = fopen(fnjson.c_str(), "r");
  if (!f) {
    LOG(INFO) << "Could not open JSON file " << fnjson;
    return std::make_tuple(std::string(""), std::string(""), -1);
  }

  // create streamer
  char readBuffer[65536];
  FileReadStream is(f, readBuffer, sizeof(readBuffer));

  // parse the json file
  Document doc;
  doc.ParseStream(is);
  auto [dfn, fmode, ntfm] = readJsonDocument(&doc);

  // clean up
  fclose(f);

  return std::make_tuple(dfn, fmode, ntfm);
}

std::tuple<std::string, std::string, int> DataOutputDirector::readJsonString(std::string const& stjson)
{
  // parse the json string
  Document doc;
  doc.Parse(stjson.c_str());
  auto [dfn, fmode, ntfm] = readJsonDocument(&doc);

  return std::make_tuple(dfn, fmode, ntfm);
}

std::tuple<std::string, std::string, int> DataOutputDirector::readJsonDocument(Document* doc)
{
  std::string smc(":");
  std::string slh("/");

  // initialisations
  std::string dfn("");
  std::string fmode("");
  int ntfm = -1;

  // is it a proper json document?
  if (!doc->HasParseError()) {

    // OutputDirector
    const Value& dod = (*doc)["OutputDirector"];
    if (dod.IsObject()) {

      // loop over the dod members
      for (auto item = dod.MemberBegin(); item != dod.MemberEnd(); ++item) {

        // get item name and value
        auto itemname = item->name.GetString();
        const Value& v = dod[itemname];

        // check possible items
        if (std::strcmp(itemname, "resfile") == 0) {
          // default result file name
          if (v.IsString()) {
            dfn = v.GetString();
            setDefaultfname(dfn);
          }
        } else if (std::strcmp(itemname, "resfilemode") == 0) {
          // open mode for result file
          if (v.IsString()) {
            fmode = v.GetString();
          }
        } else if (std::strcmp(itemname, "ntfmerge") == 0) {
          // number of time frames to merge
          if (v.IsNumber()) {
            ntfm = v.GetInt();
          }
        } else if (std::strcmp(itemname, "OutputDescriptions") == 0) {
          // array of DataOutputDescriptions
          if (v.IsArray()) {
            for (auto& od : v.GetArray()) {
              if (od.IsObject()) {
                // DataOutputDescription
                std::string s = "";
                if (od.HasMember("table")) {
                  s += od["table"].GetString();
                }
                s += smc;
                if (od.HasMember("treename")) {
                  s += od["treename"].GetString();
                }
                s += smc;
                if (od.HasMember("columns")) {
                  if (od["columns"].IsArray()) {
                    auto cs = od["columns"].GetArray();
                    for (auto& c : cs)
                      s += (c == cs[0]) ? c.GetString() : slh + c.GetString();
                  }
                }
                s += smc;
                if (od.HasMember("filename")) {
                  s += od["filename"].GetString();
                }

                // convert s to DataOutputDescription object
                readString(s);
              }
            }
          }
        }
      }
    } else {
      LOG(INFO) << "Couldn't find an OutputDirector in JSON document";
    }
  } else {
    LOG(INFO) << "Problem with JSON document";
  }

  return std::make_tuple(dfn, fmode, ntfm);
}

std::vector<DataOutputDescriptor*> DataOutputDirector::getDataOutputDescriptors(header::DataHeader dh)
{
  std::vector<DataOutputDescriptor*> result;

  // compute list of matching outputs
  data_matcher::VariableContext context;

  for (auto dodescr : dodescrs) {
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

  for (auto dodescr : dodescrs) {
    if (dodescr->matcher->match(concrete, context)) {
      result.emplace_back(dodescr);
    }
  }

  return result;
}

TFile* DataOutputDirector::getDataOutputFile(DataOutputDescriptor* dod,
                                             int ntf, int ntfmerge,
                                             std::string filemode)
{
  // initialisation
  TFile* fout = nullptr;

  // search dod->filename in fnames and return corresponding fout
  auto it = std::find(fnames.begin(), fnames.end(), dod->getFilename());
  if (it != fnames.end()) {
    int ind = std::distance(fnames.begin(), it);

    // check if new version of file needs to be opened
    int fcnt = (int)(ntf / ntfmerge);
    if ((ntf % ntfmerge) == 0 && fcnt > fcnts[ind]) {
      if (fouts[ind]) {
        fouts[ind]->Close();
      }

      fcnts[ind] = fcnt;
      auto fn = fnames[ind] + "_" + std::to_string(fcnts[ind]) + ".root";
      fouts[ind] = new TFile(fn.c_str(), filemode.c_str());
    }
    fout = fouts[ind];
  }

  return fout;
}

void DataOutputDirector::closeDataOutputFiles()
{
  for (auto fout : fouts)
    if (fout) {
      fout->Close();
    }
}

void DataOutputDirector::printOut()
{
  LOG(INFO) << "DataOutputDirector";
  LOG(INFO) << "  Default file name: " << defaultfname;
  LOG(INFO) << "  Number of dods   : " << ndod;
  LOG(INFO) << "  Number of files  : " << fnames.size();

  LOG(INFO) << "  dods:";
  for (auto const& ds : dodescrs)
    ds->printOut();

  LOG(INFO) << "  File names:";
  for (auto const& fb : fnames)
    LOG(INFO) << fb;
}

void DataOutputDirector::setDefaultfname(std::string dfn)
{
  // reset
  defaultfname = dfn;

  fnames.clear();
  tnfns.clear();
  closeDataOutputFiles();
  fouts.clear();
  fcnts.clear();

  // loop over DataOutputDescritors
  for (auto dod : dodescrs) {
    fnames.emplace_back(dod->getFilename());
    tnfns.emplace_back(dod->treename + dod->getFilename());
  }

  // the combination [tree name/file name] must be unique
  // throw exception if this is not the case
  auto it = std::unique(tnfns.begin(), tnfns.end());
  if (it != tnfns.end()) {
    printOut();
    LOG(FATAL) << "Dublicate tree names in a file!";
  }

  // make unique/sorted list of fnames
  std::sort(fnames.begin(), fnames.end());
  auto last = std::unique(fnames.begin(), fnames.end());
  fnames.erase(last, fnames.end());

  // prepare list fouts of TFile and fcnts
  for (auto fn : fnames) {
    fouts.emplace_back(new TFile());
    fcnts.emplace_back(-1);
  }
}

} // namespace framework
} // namespace o2
