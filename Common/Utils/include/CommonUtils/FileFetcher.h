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

/// \author ruben.shahoyan@cern.ch
/// Code mostly based on DataDistribution:SubTimeFrameFileSource by Gvozden Nescovic

#ifndef ALICEO2_FILEFETCHER_H_
#define ALICEO2_FILEFETCHER_H_

#include "CommonUtils/FIFO.h"
#include <unordered_map>
#include <string>
#include <thread>
#include <Rtypes.h>
#include <mutex>
#include <regex>

namespace o2
{
namespace utils
{

class FileFetcher
{
 public:
  struct FileRef {
    std::string origName{};
    std::string localName{}; // local alias for for remote files
    bool remote = false;
    bool copied = false;

    const auto& getLocalName() const { return remote ? localName : origName; }
    const auto& getOrigName() const { return origName; }
  };

  /*
  * Create file fetcher with predefined cache and async copy of remote files
  *
  * @param input:     comma-separated list of input data files and/or files with list of data files and/or directories
  * @param selRegex:  regex expression to select files needed for input
  * @param remRegex:  optional regex expression to recognize remote files
  * @param copyCmd:   optional command to copy remote files in format "<operation> ?src ?dst"
  * @param copyCmd:   base directory for copied remote files
  */
  FileFetcher(const std::string& input,
              const std::string& selRegex = "",
              const std::string& remRegex = "",
              const std::string& copyCmd = "",
              const std::string& copyDir = "/tmp");
  ~FileFetcher();

  const auto& getFileRef(size_t i) const { return mInputFiles[i]; }
  void setFailThreshold(float f) { mFailThreshold = f; }
  float getFailThreshold() const { return mFailThreshold; }
  void setMaxFilesInQueue(size_t s) { mMaxInQueue = s > 0 ? s : 1; }
  void setMaxLoops(size_t v) { mMaxLoops = v; }
  bool isRunning() const { return mRunning; }
  bool isFailed() const { return mFailure; }
  void start();
  void stop();
  void cleanup();
  size_t getNLoops() const { return mNLoops; }
  size_t getNFilesProc() const { return mNFilesProc; }
  size_t getNFilesProcOK() const { return mNFilesProcOK; }
  size_t getMaxFilesInQueue() const { return mMaxInQueue; }
  size_t getNRemoteFiles() const { return mNRemote; }
  size_t getNFiles() const { return mInputFiles.size(); }
  size_t popFromQueue(bool discard = false);
  size_t getQueueSize() const { return mQueue.size(); }
  std::string getNextFileInQueue() const;
  void discardFile(const std::string& fname);

 private:
  size_t nextInQueue() const;
  void processInput(const std::string& input);
  void processInput(const std::vector<std::string>& input);
  void processDirectory(const std::string& name);
  bool addInputFile(const std::string& fname);
  std::string createCopyName(const std::string& fname) const;
  bool copyFile(size_t id);
  bool isRemote(const std::string& fname) const;
  void fetcher();

 private:
  FIFO<size_t> mQueue{};
  std::string mCopyDirName{"/tmp"};
  std::string mCopyCmdLogFile{};
  std::string mCopyCmd{};
  std::unique_ptr<std::regex> mSelRegex;
  std::unique_ptr<std::regex> mRemRegex;
  std::unordered_map<std::string, size_t> mCopied{};
  std::vector<FileRef> mInputFiles{};
  size_t mNRemote{0};
  size_t mMaxInQueue{5};
  bool mRunning = false;
  bool mNoRemoteCopy = false;
  bool mFailure = false;
  size_t mMaxLoops = 0;
  size_t mNLoops = 0;
  size_t mNFilesProc = 0;
  size_t mNFilesProcOK = 0;
  float mFailThreshold = 0.f; // throw if too many failed fetches (>0 : fraction to total, <0 abs number)
  mutable std::mutex mMtx;
  std::mutex mMtxStop;
  std::thread mFetcherThread{};

  ClassDefNV(FileFetcher, 1);
};

} // namespace utils
} // namespace o2

#endif
