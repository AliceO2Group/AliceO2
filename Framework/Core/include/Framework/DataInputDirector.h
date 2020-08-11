// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_DataInputDirector_H_INCLUDED
#define o2_framework_DataInputDirector_H_INCLUDED

#include "TFile.h"
#include "TTreeReader.h"

#include "Framework/DataDescriptorMatcher.h"

#include <regex>
#include "rapidjson/fwd.h"

namespace o2
{
namespace framework
{
using namespace rapidjson;

struct DataInputDescriptor {
  /// Holds information concerning the reading of an aod table.
  /// The information includes the table specification, treename,
  /// and the input files

  std::string tablename = "";
  std::string treename = "";
  std::unique_ptr<data_matcher::DataDescriptorMatcher> matcher;

  DataInputDescriptor() = default;
  DataInputDescriptor(bool alienSupport);

  void printOut();

  // setters
  void setInputfilesFile(std::string dffn) { minputfilesFile = dffn; }
  void setInputfilesFile(std::string* dffnptr) { minputfilesFilePtr = dffnptr; }
  void setFilenamesRegex(std::string fn) { mFilenameRegex = fn; }
  void setFilenamesRegex(std::string* fnptr) { mFilenameRegexPtr = fnptr; }

  void setDefaultInputfiles(std::vector<std::string>* difnptr) { mdefaultFilenamesPtr = difnptr; }

  void addFilename(std::string fn);
  int fillInputfiles();

  // getters
  std::string getInputfilesFilename();
  std::string getFilenamesRegexString();
  std::regex getFilenamesRegex();
  int getNumberInputfiles() { return mfilenames.size(); }

  TFile* getInputFile(int counter);
  void closeInputFile();
  std::string getInputFilename(int counter);
  bool isAlienSupportOn() { return mAlienSupport; }

 private:
  std::string minputfilesFile = "";
  std::string* minputfilesFilePtr = nullptr;
  std::string mFilenameRegex = "";
  std::string* mFilenameRegexPtr = nullptr;
  std::vector<std::string> mfilenames;
  std::vector<std::string>* mdefaultFilenamesPtr = nullptr;
  TFile* mcurrentFile = nullptr;
  bool mAlienSupport = false;
};

struct DataInputDirector {
  /// Holds a list of DataInputDescriptor
  /// Provides functionality to access the matching DataInputDescriptor
  /// and the related input files

  DataInputDirector();
  DataInputDirector(std::string inputFile);

  void reset();
  void createDefaultDataInputDescriptor();
  void printOut();
  bool atEnd(int counter);

  // setters
  void setInputfilesFile(std::string iffn) { minputfilesFile = iffn; }
  void setFilenamesRegex(std::string dfn) { mFilenameRegex = dfn; }
  bool readJson(std::string const& fnjson);
  void closeInputFiles();

  // getters
  DataInputDescriptor* getDataInputDescriptor(header::DataHeader dh);
  std::unique_ptr<TTreeReader> getTreeReader(header::DataHeader dh, int counter, std::string treeName);
  std::string getInputFilename(header::DataHeader dh, int counter);
  TTree* getDataTree(header::DataHeader dh, int counter);
  int getNumberInputDescriptors() { return mdataInputDescriptors.size(); }

 private:
  std::string minputfilesFile;
  std::string* const minputfilesFilePtr = &minputfilesFile;
  std::string mFilenameRegex;
  std::string* const mFilenameRegexPtr = &mFilenameRegex;
  DataInputDescriptor* mdefaultDataInputDescriptor = nullptr;
  std::vector<std::string> mdefaultInputFiles;
  std::vector<DataInputDescriptor*> mdataInputDescriptors;

  bool mDebugMode = false;
  bool mAlienSupport = false;

  bool readJsonDocument(Document* doc);
  bool isValid();
};

} // namespace framework
} // namespace o2

#endif // o2_framework_DataInputDirector_H_INCLUDED
