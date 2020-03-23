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

namespace o2
{
namespace framework
{
namespace data_matcher
{

DataOutputDirector::DataOutputDirector()
{
  defaultfname = std::string("AnalysisResults");
}

void DataOutputDirector::reset()
{
  dodescrs.clear();
  fnames.clear();
  closeDataOutputFiles();
  fouts.clear();
  fcnts.clear();
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
    if (dod->filename.empty())
      dod->setFilename(defaultfname);

    dodescrs.emplace_back(dod);
    fnames.emplace_back(dod->filename);
    tnfns.emplace_back(dod->treename + dod->filename);
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

std::vector<DataOutputDescriptor*> DataOutputDirector::getDataOutputDescriptors(header::DataHeader dh)
{
  std::vector<DataOutputDescriptor*> result;

  // compute list of matching outputs
  VariableContext context;

  for (auto dodescr : dodescrs) {
    if (dodescr->matcher->match(dh, context))
      result.emplace_back(dodescr);
  }

  return result;
}

std::vector<DataOutputDescriptor*> DataOutputDirector::getDataOutputDescriptors(InputSpec spec)
{
  std::vector<DataOutputDescriptor*> result;

  // compute list of matching outputs
  VariableContext context;
  auto concrete = std::get<ConcreteDataMatcher>(spec.matcher);

  for (auto dodescr : dodescrs) {
    if (dodescr->matcher->match(concrete, context))
      result.emplace_back(dodescr);
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
  auto it = std::find(fnames.begin(), fnames.end(), dod->filename);
  if (it != fnames.end()) {
    int ind = std::distance(fnames.begin(), it);

    // check if new version of file needs to be opened
    int fcnt = (int)(ntf / ntfmerge);
    if ((ntf % ntfmerge) == 0 && fcnt > fcnts[ind]) {
      if (fouts[ind])
        fouts[ind]->Close();

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
    if (fout)
      fout->Close();
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

} // namespace data_matcher
} // namespace framework
} // namespace o2
