// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TimeframeValidatorDevice.cxx
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-07
/// @brief  Validator device for a full time frame

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "DataFlow/TimeframeWriterDevice.h"
#include "DataFlow/TimeframeParser.h"
#include "TimeFrame/TimeFrame.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/DataHeader.h"
#include <options/FairMQProgOptions.h>
#include <boost/filesystem.hpp>


using DataHeader = o2::header::DataHeader;
using IndexElement = o2::DataFormat::IndexElement;

namespace o2 { namespace DataFlow {

TimeframeWriterDevice::TimeframeWriterDevice()
  : O2Device{}
  , mInChannelName{}
  , mFile{}
  , mMaxTimeframes{}
  , mMaxFileSize{}
  , mMaxFiles{}
  , mFileCount{0}
{
}

void TimeframeWriterDevice::InitTask()
{
  mInChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
  mOutFileName = GetConfig()->GetValue<std::string>(OptionKeyOutputFileName);
  mMaxTimeframes = GetConfig()->GetValue<size_t>(OptionKeyMaxTimeframesPerFile);
  mMaxFileSize = GetConfig()->GetValue<size_t>(OptionKeyMaxFileSize);
  mMaxFiles = GetConfig()->GetValue<size_t>(OptionKeyMaxFiles);
}

void TimeframeWriterDevice::Run()
{
  boost::filesystem::path p(mOutFileName);
  size_t streamedTimeframes = 0;
  bool needsNewFile = true;
  while (CheckCurrentState(RUNNING) && mFileCount < mMaxFiles) {
    // In case we need to process more than one file,
    // the filename is split in basename and extension
    // and we call the files `<basename><count>.<extension>`.
    if (needsNewFile) {
      std::string filename = mOutFileName;
      if (mMaxFiles > 1) {
        std::string base_path(mOutFileName,  0, mOutFileName.find_last_of("."));
        std::string extension(mOutFileName,  mOutFileName.find_last_of("."));
        filename = base_path + std::to_string(mFileCount) + extension;
      }
      LOG(INFO) << "Opening " << filename << " for output\n";
      mFile.open(filename.c_str(), std::ofstream::out | std::ofstream::binary);
      needsNewFile = false;
    }

    FairMQParts timeframeParts;
    if (Receive(timeframeParts, mInChannelName, 0, 100) <= 0)
      continue;

    streamTimeframe(mFile, timeframeParts);
    if ((mFile.tellp() > mMaxFileSize) || (streamedTimeframes++ > mMaxTimeframes))
    {
      mFile.flush();
      mFile.close();
      mFileCount++;
      needsNewFile = true;
    }
  }
}

void TimeframeWriterDevice::PostRun()
{
  if (mFile.is_open()) {
    mFile.flush();
    mFile.close();
  }
}

}} // namespace o2::DataFlow
