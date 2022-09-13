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
#ifndef o2_framework_DataInputDirector_H_INCLUDED
#define o2_framework_DataInputDirector_H_INCLUDED

#include "TFile.h"
#include "TTreeReader.h"

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataAllocator.h"
#include "Monitoring/Monitoring.h"

#include <regex>
#include "rapidjson/fwd.h"

namespace o2::framework
{

struct FileNameHolder {
  std::string fileName;
  int numberOfTimeFrames = 0;
  std::vector<uint64_t> listOfTimeFrameNumbers;
  std::vector<std::string> listOfTimeFrameKeys;
  std::vector<bool> alreadyRead;
};
FileNameHolder* makeFileNameHolder(std::string fileName);

struct FileAndFolder {
  TFile* file = nullptr;
  std::string folderName = "";
};

class DataInputDescriptor
{
  /// Holds information concerning the reading of an aod table.
  /// The information includes the table specification, treename,
  /// and the input files

 public:
  std::string tablename = "";
  std::string treename = "";
  std::unique_ptr<data_matcher::DataDescriptorMatcher> matcher;

  DataInputDescriptor() = default;
  DataInputDescriptor(bool alienSupport, int level, o2::monitoring::Monitoring* monitoring = nullptr, std::string parentFileReplacement = "");

  void printOut();

  // setters
  void setInputfilesFile(std::string dffn) { minputfilesFile = dffn; }
  void setInputfilesFile(std::string* dffnptr) { minputfilesFilePtr = dffnptr; }
  void setFilenamesRegex(std::string fn) { mFilenameRegex = fn; }
  void setFilenamesRegex(std::string* fnptr) { mFilenameRegexPtr = fnptr; }

  void setDefaultInputfiles(std::vector<FileNameHolder*>* difnptr) { mdefaultFilenamesPtr = difnptr; }

  void addFileNameHolder(FileNameHolder* fn);
  int fillInputfiles();
  bool setFile(int counter);

  // getters
  std::string getInputfilesFilename();
  std::string getFilenamesRegexString();
  std::regex getFilenamesRegex();
  int getNumberInputfiles() { return mfilenames.size(); }
  int getNumberTimeFrames() { return mtotalNumberTimeFrames; }
  int findDFNumber(int file, std::string dfName);

  uint64_t getTimeFrameNumber(int counter, int numTF);
  FileAndFolder getFileFolder(int counter, int numTF);
  DataInputDescriptor* getParentFile(int counter, int numTF);
  int getTimeFramesInFile(int counter);
  int getReadTimeFramesInFile(int counter);

  bool readTree(DataAllocator& outputs, header::DataHeader dh, int counter, int numTF, std::string treename, size_t& totalSizeCompressed, size_t& totalSizeUncompressed);

  void printFileStatistics();
  void closeInputFile();
  bool isAlienSupportOn() { return mAlienSupport; }

 private:
  std::string minputfilesFile = "";
  std::string* minputfilesFilePtr = nullptr;
  std::string mFilenameRegex = "";
  std::string* mFilenameRegexPtr = nullptr;
  std::string mParentFileReplacement;
  std::vector<FileNameHolder*> mfilenames;
  std::vector<FileNameHolder*>* mdefaultFilenamesPtr = nullptr;
  TFile* mcurrentFile = nullptr;
  int mCurrentFileID = -1;
  bool mAlienSupport = false;

  o2::monitoring::Monitoring* mMonitoring = nullptr;

  TMap* mParentFileMap = nullptr;
  DataInputDescriptor* mParentFile = nullptr;
  int mLevel = 0; // level of parent files

  int mtotalNumberTimeFrames = 0;

  uint64_t mIOTime = 0;
  uint64_t mCurrentFileStartedAt = 0;
};

class DataInputDirector
{
  /// Holds a list of DataInputDescriptor
  /// Provides functionality to access the matching DataInputDescriptor
  /// and the related input files

 public:
  DataInputDirector();
  DataInputDirector(std::string inputFile, o2::monitoring::Monitoring* monitoring = nullptr, std::string parentFileReplacement = "");
  DataInputDirector(std::vector<std::string> inputFiles, o2::monitoring::Monitoring* monitoring = nullptr, std::string parentFileReplacement = "");
  ~DataInputDirector();

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
  int getNumberInputDescriptors() { return mdataInputDescriptors.size(); }

  std::unique_ptr<TTreeReader> getTreeReader(header::DataHeader dh, int counter, int numTF, std::string treeName);
  bool readTree(DataAllocator& outputs, header::DataHeader dh, int counter, int numTF, size_t& totalSizeCompressed, size_t& totalSizeUncompressed);
  uint64_t getTimeFrameNumber(header::DataHeader dh, int counter, int numTF);
  FileAndFolder getFileFolder(header::DataHeader dh, int counter, int numTF);
  int getTimeFramesInFile(header::DataHeader dh, int counter);

  uint64_t getTotalSizeCompressed();
  uint64_t getTotalSizeUncompressed();

 private:
  std::string minputfilesFile;
  std::string* const minputfilesFilePtr = &minputfilesFile;
  std::string mFilenameRegex;
  std::string mParentFileReplacement;
  std::string* const mFilenameRegexPtr = &mFilenameRegex;
  DataInputDescriptor* mdefaultDataInputDescriptor = nullptr;
  std::vector<FileNameHolder*> mdefaultInputFiles;
  std::vector<DataInputDescriptor*> mdataInputDescriptors;

  o2::monitoring::Monitoring* mMonitoring = nullptr;

  bool mDebugMode = false;
  bool mAlienSupport = false;

  bool readJsonDocument(rapidjson::Document* doc);
  bool isValid();
};

} // namespace o2::framework

#endif // o2_framework_DataInputDirector_H_INCLUDED
