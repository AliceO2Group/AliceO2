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

/// @author Christian Holm Christensen <cholm@nbi.dk>

#include "SimulationDataFormat/MCUtils.h"
#include "Generators/GeneratorFileOrCmd.h"
// For fifo's and system call
#include <cstdlib>
#include <sys/types.h> // POSIX only
#include <sys/stat.h>  // POISX only
#include <cstdio>
// For filesystem operations
#include <filesystem>
// Waits
#include <thread>
// Log messages
#include <fairlogger/Logger.h>
#include <SimConfig/SimConfig.h>

namespace
{
std::string ltrim(const std::string& s, const std::string& what = "\" ' ")
{
  auto start = s.find_first_not_of(what);
  if (start == std::string::npos) {
    return "";
  }

  return s.substr(start);
}

std::string rtrim(const std::string& s, const std::string& what = "\"' ")
{
  auto end = s.find_last_not_of(what);
  return s.substr(0, end + 1);
}

std::string trim(const std::string& s, const std::string& what = "\"' ")
{
  return rtrim(ltrim(s, what), what);
}
} // namespace

namespace o2
{
namespace eventgen
{
// -----------------------------------------------------------------
void GeneratorFileOrCmd::setup(const GeneratorFileOrCmdParam& param,
                               const conf::SimConfig& config)
{
  setFileNames(param.fileNames);
  setCmd(param.cmd);
  setOutputSwitch(trim(param.outputSwitch));
  setSeedSwitch(trim(param.seedSwitch));
  setBmaxSwitch(trim(param.bMaxSwitch));
  setNEventsSwitch(trim(param.nEventsSwitch));
  setBackgroundSwitch(trim(param.backgroundSwitch));
  setSeed(config.getStartSeed());
  setNEvents(config.getNEvents());
  setBmax(config.getBMax());
}
// -----------------------------------------------------------------
void GeneratorFileOrCmd::setFileNames(const std::string& filenames)
{
  std::stringstream s(filenames);
  std::string f;
  while (std::getline(s, f, ',')) {
    mFileNames.push_back(f);
  }
}
// -----------------------------------------------------------------
std::string GeneratorFileOrCmd::makeCmdLine() const
{
  std::string fileName = mFileNames.front();
  std::stringstream s;
  s << mCmd << " ";
  if (not mSeedSwitch.empty() and mSeedSwitch != "none") {
    s << mSeedSwitch << " " << mSeed << " ";
  }
  if (not mNEventsSwitch.empty() and mNEventsSwitch != "none") {
    s << mNEventsSwitch << " " << mNEvents << " ";
  }
  if (not mBmaxSwitch.empty() and mBmax >= 0 and mBmaxSwitch != "none") {
    s << mBmaxSwitch.c_str() << " " << mBmax << " ";
  }

  s << mOutputSwitch << " " << fileName << " "
    << mBackgroundSwitch;
  return s.str();
}
// -----------------------------------------------------------------
bool GeneratorFileOrCmd::executeCmdLine(const std::string& cmd) const
{
  LOG(info) << "Command line to execute: \"" << cmd << "\"";
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    LOG(fatal) << "Failed to spawn \"" << cmd << "\"";
    return false;
  }
  return true;
}
// -----------------------------------------------------------------
bool GeneratorFileOrCmd::makeTemp()
{
  mFileNames.clear();
  char buf[] = "generatorFifoXXXXXX";
  auto fp = mkstemp(buf);
  if (fp < 0) {
    LOG(fatal) << "Failed to make temporary file: "
               << std::strerror(errno);
    return false;
  }
  mTemporary = std::string(buf);
  mFileNames.push_back(mTemporary);
  close(fp);
  return true;
}
// -----------------------------------------------------------------
bool GeneratorFileOrCmd::removeTemp() const
{
  if (mTemporary.empty()) {
    LOG(info) << "Temporary file name empty, nothing to remove";
    return false;
  }

  // Get the file we're reading from
  std::filesystem::path p(mTemporary);

  // Check if the file exists
  if (not std::filesystem::exists(p)) {
    LOG(info) << "Temporary file " << p << " does not exist";
    return true;
  }

  // Remove temporary file
  std::error_code ec;
  std::filesystem::remove(p, ec);
  if (ec) {
    LOG(warn) << "When removing " << p << ": " << ec.message();
  }

  // Ignore errors when removing the temporary file
  return true;
}
// -----------------------------------------------------------------
bool GeneratorFileOrCmd::makeFifo() const
{
  // First remove the temporary file if it exists,
  // otherwise we may not be able to make the FIFO
  removeTemp();

  std::string fileName = mFileNames.front();

  int ret = mkfifo(fileName.c_str(), 0600);
  if (ret != 0) {
    LOG(fatal) << "Failed to make fifo \"" << fileName << "\": "
               << std::strerror(errno);
    return false;
  }

  return true;
}
// -----------------------------------------------------------------
bool GeneratorFileOrCmd::ensureFiles()
{
  try {
    for (auto& f : mFileNames) {
      auto c = std::filesystem::canonical(std::filesystem::path(f));
      f = c.c_str();
    }
  } catch (std::exception& e) {
    LOG(error) << e.what();
    return false;
  }
  return true;
}
// -----------------------------------------------------------------
void GeneratorFileOrCmd::waitForData(const std::string& filename) const
{
  using namespace std::chrono_literals;

  // Get the file we're reading from
  std::filesystem::path p(filename);

  LOG(debug) << "Waiting for data on " << p;

  // Wait until child process creates the file
  while (not std::filesystem::exists(p)) {
    std::this_thread::sleep_for(mWait * 1ms);
  }

  // Wait until we have more data in the file than just the file
  // header
  while (std::filesystem::file_size(p) <= 256) {
    std::this_thread::sleep_for(mWait * 1ms);
  }

  // Give the child process 1 second to post the data to the file
  LOG(debug) << "Got data in " << p << ", sleeping for a while";
  std::this_thread::sleep_for(mWait * 2ms);
}

} /* namespace eventgen */
} /* namespace o2 */
