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
/// Code partially based on DataDistribution:SubTimeFrameFileSource by Gvozden Nescovic

#include "CommonUtils/FileFetcher.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/FileSystemUtils.h"
#include "Framework/Logger.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <locale>
#include <TGrid.h>
#include <TSystem.h>

using namespace o2::utils;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

//____________________________________________________________
FileFetcher::FileFetcher(const std::string& input, const std::string& selRegex, const std::string& remRegex,
                         const std::string& copyCmd, const std::string& copyDir)
  : mCopyCmd(copyCmd), mCopyDirName(copyDir)
{
  if (!selRegex.empty()) {
    mSelRegex = std::make_unique<std::regex>(selRegex.c_str());
  }
  if (!remRegex.empty()) {
    mRemRegex = std::make_unique<std::regex>(remRegex);
  }
  mNoRemoteCopy = mCopyCmd == "no-copy";

  // parse input list
  mCopyDirName = o2::utils::Str::create_unique_path(mCopyDirName, 8);
  processInput(input);
  LOGP(info, "Input contains {} files, {} remote", getNFiles(), mNRemote);
  if (mNRemote) {
    if (mNoRemoteCopy) { // make sure the copy command is provided, unless copy was explicitly forbidden
      LOGP(info, "... but their local copying is explicitly forbidden");
    } else {
      if (mCopyCmd.find("?src") == std::string::npos || mCopyCmd.find("?dst") == std::string::npos) {
        throw std::runtime_error(fmt::format("remote files asked but copy cmd \"{}\" is not valid", mCopyCmd));
      }
      try {
        o2::utils::createDirectoriesIfAbsent(mCopyDirName);
      } catch (...) {
        throw std::runtime_error(fmt::format("failed to create scratch directory {}", mCopyDirName));
      }
      mCopyCmdLogFile = fmt::format("{}/{}", mCopyDirName, "copy-cmd.log");
      LOGP(info, "FileFetcher tmp scratch directory is set to {}", mCopyDirName);
    }
  }
}

//____________________________________________________________
FileFetcher::~FileFetcher()
{
  stop();
  cleanup();
}

//____________________________________________________________
void FileFetcher::processInput(const std::string& input)
{
  auto ftokens = o2::utils::Str::tokenize(input, ',', true); // in case multiple inputs are provided
  processInput(ftokens);
}

//____________________________________________________________
void FileFetcher::processInput(const std::vector<std::string>& input)
{
  for (auto inp : input) {
    o2::utils::Str::trim(inp);

    if (fs::is_directory(inp)) {
      processDirectory(inp);
    } else if (mSelRegex && !std::regex_match(inp, *mSelRegex.get())) { // provided selector does not match, treat as a txt file with list
      std::ifstream listFile(inp);
      if (!listFile.good()) {
        LOGP(error, "file {} pretends to be a list of inputs but does not exist", inp);
        continue;
      }
      std::string line;
      std::vector<std::string> newInput;
      while (getline(listFile, line)) {
        o2::utils::Str::trim(line);
        if (line[0] == '#') { // ignore commented file
          continue;
        }
        newInput.push_back(line);
      }
      processInput(newInput);
    } else { // should be local or remote data file
      addInputFile(inp);
    }
  }
}

//____________________________________________________________
void FileFetcher::processDirectory(const std::string& name)
{
  std::vector<std::string> vs;
  for (auto const& entry : fs::directory_iterator{name}) {
    const auto& fnm = entry.path().native();
    if (fs::is_regular_file(fnm) && (!mSelRegex || std::regex_match(fnm, *mSelRegex.get()))) {
      vs.push_back(fnm);
    }
  }
  std::sort(vs.begin(), vs.end());
  for (const auto& s : vs) {
    addInputFile(s); // local files only
  }
}

//____________________________________________________________
bool FileFetcher::addInputFile(const std::string& fname)
{
  static bool alienErrorPrinted = false;
  if (mRemRegex && std::regex_match(fname, *mRemRegex.get())) {
    mInputFiles.emplace_back(FileRef{fname, mNoRemoteCopy ? fname : createCopyName(fname), true, false});
    if (fname.find("alien:") == 0) {
      if (!gGrid && !TGrid::Connect("alien://") && !alienErrorPrinted) {
        LOG(error) << "File name starts with alien but connection to Grid failed";
        alienErrorPrinted = true;
      }
    }
    mNRemote++;
  } else if (fs::exists(fname)) { // local file
    mInputFiles.emplace_back(FileRef{fname, "", false, false});
  } else {
    LOGP(error, "file {} pretends to be local but does not exist", fname);
    return false;
  }
  return true;
}

//____________________________________________________________
std::string FileFetcher::createCopyName(const std::string& fname) const
{
  std::string cpnam{}, cpnamP = fname;
  for (auto& c : cpnamP) {
    if (!std::isalnum(c) && c != '.' && c != '-') {
      c = '_';
    }
  }
  while (1) {
    cpnam = fmt::format("{}/{}_{}", mCopyDirName, o2::utils::Str::getRandomString(12), cpnamP);
    if (!fs::exists(cpnam)) {
      break;
    }
  }
  return cpnam;
}

//____________________________________________________________
size_t FileFetcher::popFromQueue(bool discard)
{
  // remove file from the queue, if requested and if it was copied, remove copy
  std::lock_guard<std::mutex> lock(mMtx);
  const auto* ptr = mQueue.frontPtr();
  if (mQueue.empty()) {
    return -1ul;
  }
  auto id = mQueue.front();
  mQueue.pop();
  if (discard) {
    discardFile(mInputFiles[id].getLocalName());
  }
  return id;
}

//____________________________________________________________
size_t FileFetcher::nextInQueue() const
{
  return mQueue.empty() ? -1ul : mQueue.front();
}

//____________________________________________________________
std::string FileFetcher::getNextFileInQueue() const
{
  if (mQueue.empty()) {
    return {};
  }
  return mQueue.empty() ? "" : mInputFiles[mQueue.front()].getLocalName();
}

//____________________________________________________________
void FileFetcher::start()
{
  if (mRunning) {
    return;
  }
  mRunning = true;
  mFetcherThread = std::thread(&FileFetcher::fetcher, this);
}

//____________________________________________________________
void FileFetcher::stop()
{
  mRunning = false;
  std::lock_guard<std::mutex> lock(mMtxStop);
  if (mFetcherThread.joinable()) {
    mFetcherThread.join();
  }
}

//____________________________________________________________
void FileFetcher::cleanup()
{
  if (mRunning) {
    throw std::runtime_error("FileFetcher thread is still active, cannot cleanup");
  }
  if (mNRemote && o2::utils::Str::pathExists(mCopyDirName)) {
    try {
      fs::remove_all(mCopyDirName);
    } catch (...) {
      LOGP(error, "FileFetcher failed to remove sctrach directory {}", mCopyDirName);
    }
  }
}

//____________________________________________________________
void FileFetcher::fetcher()
{
  // data fetching/copying thread
  size_t fileEntry = -1ul;

  if (!getNFiles()) {
    mRunning = false;
    return;
  }

  // BOOST requires a locale set
  try {
    std::locale loc("");
  } catch (const std::exception& e) {
    setenv("LC_ALL", "C", 1);
    try {
      std::locale loc("");
      LOG(info) << "Setting locale";
    } catch (const std::exception& e) {
      LOG(info) << "Setting locale failed: " << e.what();
      return;
    }
  }

  while (mRunning) {
    mNLoops = mNFilesProc / getNFiles();
    if (mNLoops > mMaxLoops) {
      LOGP(info, "Finished file fetching: {} of {} files fetched successfully in {} iterations", mNFilesProcOK, mNFilesProc, mMaxLoops);
      mRunning = false;
      break;
    }
    if (getQueueSize() >= mMaxInQueue) {
      std::this_thread::sleep_for(5ms);
      continue;
    }
    fileEntry = (fileEntry + 1) % getNFiles();
    if (fileEntry == 0 && mNLoops > 0) {
      LOG(info) << "Fetcher starts new iteration " << mNLoops;
    }
    mNFilesProc++;
    auto& fileRef = mInputFiles[fileEntry];
    if (fileRef.copied || !fileRef.remote || mNoRemoteCopy) {
      mQueue.push(fileEntry);
      mNFilesProcOK++;
    } else { // need to copy
      if (copyFile(fileEntry)) {
        fileRef.copied = true;
        mQueue.push(fileEntry);
        mNFilesProcOK++;
      }
    }
  }
}

//____________________________________________________________
void FileFetcher::discardFile(const std::string& fname)
{
  // delete file if it is copied.
  auto ent = mCopied.find(fname);
  if (ent != mCopied.end()) {
    mInputFiles[ent->second - 1].copied = false;
    fs::remove(fname);
    mCopied.erase(fname);
  }
}

//____________________________________________________________
bool FileFetcher::copyFile(size_t id)
{
  // copy remote file to local setCopyDirName. Adaptation for Gvozden's code from SubTimeFrameFileSource::DataFetcherThread()
  if (mCopyCmd.find("alien") != std::string::npos) {
    if (!gGrid && !TGrid::Connect("alien://")) {
      LOG(error) << "Copy command refers to alien but connection to Grid failed";
    }
  }
  auto realCmd = std::regex_replace(std::regex_replace(mCopyCmd, std::regex("\\?src"), mInputFiles[id].getOrigName()), std::regex("\\?dst"), mInputFiles[id].getLocalName());
  auto fullCmd = fmt::format("sh -c \"{}\" > {}  2>&1", realCmd, mCopyCmdLogFile);
  LOG(info) << "Executing " << fullCmd;
  const auto sysRet = gSystem->Exec(fullCmd.c_str());
  if (sysRet != 0) {
    LOGP(warning, "FileFetcher: non-zero exit code {} for cmd={}", sysRet, realCmd);
  }
  if (!fs::is_regular_file(mInputFiles[id].getLocalName()) || fs::is_empty(mInputFiles[id].getLocalName())) {
    LOGP(alarm, "FileFetcher: failed for copy command {}", realCmd);
    return false;
  }
  mCopied[mInputFiles[id].getLocalName()] = id + 1;
  return true;
}
