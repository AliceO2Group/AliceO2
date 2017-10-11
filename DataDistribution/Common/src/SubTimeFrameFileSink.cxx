// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Common/SubTimeFrameFileSink.h"
#include "Common/FilePathUtils.h"

#include <boost/program_options/options_description.hpp>
#include <boost/filesystem.hpp>

#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>

namespace o2
{
namespace DataDistribution
{

namespace bpo = boost::program_options;

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileSink
////////////////////////////////////////////////////////////////////////////////

void SubTimeFrameFileSink::start()
{
  if (enabled())
    mSinkThread = std::thread(&SubTimeFrameFileSink::DataHandlerThread, this, 0);
}

void SubTimeFrameFileSink::stop()
{
  if (mSinkThread.joinable())
    mSinkThread.join();
}

bpo::options_description SubTimeFrameFileSink::getProgramOptions()
{
  bpo::options_description lSinkDesc("(Sub)TimeFrame file sink options", 120);

  lSinkDesc.add_options()(
    OptionKeyStfSinkEnable,
    bpo::bool_switch()->default_value(false),
    "Enable writing of (Sub)TimeFrames to file.")(
    OptionKeyStfSinkDir,
    bpo::value<std::string>()->default_value(""),
    "Specifies a destination directory where (Sub)TimeFrames are to be written. "
    "Note: A new directory will be created here for all output files.")(
    OptionKeyStfSinkFileName,
    bpo::value<std::string>()->default_value("%n"),
    "Specifies file name pattern: %n - file index, %D - date, %T - time.")(
    OptionKeyStfSinkStfsPerFile,
    bpo::value<std::uint64_t>()->default_value(1),
    "Specifies number of (Sub)TimeFrames per file.")(
    OptionKeyStfSinkFileSize,
    bpo::value<std::uint64_t>()->default_value(std::uint64_t(4) << 30), /* 4GiB */
    "Specifies target size for (Sub)TimeFrame files.")(
    OptionKeyStfSinkSidecar,
    bpo::bool_switch()->default_value(false),
    "Write a sidecar file for each (Sub)TimeFrame file containing information about data blocks "
    "written in the data file. "
    "Note: Useful for debugging. "
    "Warning: sidecar file format is not stable.");

  return lSinkDesc;
}

bool SubTimeFrameFileSink::loadVerifyConfig(const FairMQProgOptions& pFMQProgOpt)
{
  mEnabled = pFMQProgOpt.GetValue<bool>(OptionKeyStfSinkEnable);

  LOG(INFO) << "(Sub)TimeFrame file sink " << (mEnabled ? "enabled" : "disabled");

  if (!mEnabled)
    return true;

  mRootDir = pFMQProgOpt.GetValue<std::string>(OptionKeyStfSinkDir);
  if (mRootDir.length() == 0) {
    LOG(ERROR) << "(Sub)TimeFrame file sink directory must be specified";
    return false;
  }

  mFileNamePattern = pFMQProgOpt.GetValue<std::string>(OptionKeyStfSinkFileName);
  mStfsPerFile = std::max(std::uint64_t(1), pFMQProgOpt.GetValue<std::uint64_t>(OptionKeyStfSinkStfsPerFile));
  mFileSize = std::max(std::uint64_t(1), pFMQProgOpt.GetValue<std::uint64_t>(OptionKeyStfSinkFileSize));
  mSidecar = pFMQProgOpt.GetValue<bool>(OptionKeyStfSinkSidecar);

  // make sure directory exists and it is writable
  namespace bfs = boost::filesystem;
  bfs::path lDirPath(mRootDir);
  if (!bfs::is_directory(lDirPath)) {
    LOG(ERROR) << "(Sub)TimeFrame file sink directory does not exist";
    return false;
  }

  // make a session directory
  mCurrentDir = (bfs::path(mRootDir) / FilePathUtils::getNextSeqName(mRootDir)).string();
  if (!bfs::create_directory(mCurrentDir)) {
    LOG(ERROR) << "Directory '" << mCurrentDir << "' for (Sub)TimeFrame file sink cannot be created";
    return false;
  }

  // print options
  LOG(INFO) << "(Sub)TimeFrame Sink :: enabled       = " << (mEnabled ? "yes" : "no");
  LOG(INFO) << "(Sub)TimeFrame Sink :: root dir      = " << mRootDir;
  LOG(INFO) << "(Sub)TimeFrame Sink :: file pattern  = " << mFileNamePattern;
  LOG(INFO) << "(Sub)TimeFrame Sink :: stfs per file = " << mStfsPerFile;
  LOG(INFO) << "(Sub)TimeFrame Sink :: max file size = " << mFileSize;
  LOG(INFO) << "(Sub)TimeFrame Sink :: sidecar files = " << (mSidecar ? "yes" : "no");
  LOG(INFO) << "(Sub)TimeFrame Sink :: write dir     = " << mCurrentDir;

  return true;
}

std::string SubTimeFrameFileSink::newStfFileName()
{
  time_t lNow;
  time(&lNow);
  char lTimeBuf[32];

  std::string lFileName = mFileNamePattern;
  std::stringstream lIdxString;
  lIdxString << std::dec << std::setw(8) << std::setfill('0') << mCurrentFileIdx;
  boost::replace_all(lFileName, "%n", lIdxString.str());

  strftime(lTimeBuf, sizeof(lTimeBuf), "%F", localtime(&lNow));
  boost::replace_all(lFileName, "%D", lTimeBuf);

  strftime(lTimeBuf, sizeof(lTimeBuf), "%H_%M_%S", localtime(&lNow));
  boost::replace_all(lFileName, "%T", lTimeBuf);

  mCurrentFileIdx++;
  return lFileName;
}

/// File writing thread
void SubTimeFrameFileSink::DataHandlerThread(const unsigned pIdx)
{
  std::uint64_t lCurrentFileSize = 0;
  std::uint64_t lCurrentFileStfs = 0;

  while (mDeviceI.CheckCurrentState(O2Device::RUNNING)) {
    // Get the next STF
    std::unique_ptr<SubTimeFrame> lStf = mPipelineI.dequeue(mPipelineStageIn);
    if (!lStf) {
      // input queue is stopped, bail out
      break;
    }

    if (!enabled()) {
      LOG(ERROR) << "Pipeline error, disabled file sing receiving STFs";
      break;
    }

    // check if we need a writer
    if (!mStfWriter) {
      namespace bfs = boost::filesystem;
      mStfWriter = std::make_unique<SubTimeFrameFileWriter>(
        bfs::path(mCurrentDir) / bfs::path(newStfFileName()),
        mSidecar);
    }

    // write
    if (mStfWriter->write(*lStf)) {
      lCurrentFileStfs++;
      lCurrentFileSize = mStfWriter->size();
    } else {
      mStfWriter.reset();
      mEnabled = false;
      LOG(ERROR) << "(Sub)TimeFrame file sink: error while writing a file";
      LOG(ERROR) << "(Sub)TimeFrame file sink: disabling writing";
    }

    // check if we should rotate the file
    if ((lCurrentFileStfs >= mStfsPerFile) || (lCurrentFileSize >= mFileSize)) {
      lCurrentFileStfs = 0;
      lCurrentFileSize = 0;
      mStfWriter.reset(nullptr);
    }

    mPipelineI.queue(mPipelineStageOut, std::move(lStf));
  }
  LOG(INFO) << "Exiting file sink thread[" << pIdx << "]...";
}
}
} /* o2::DataDistribution */
