// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TIMEFRAME_WRITER_DEVICE_H_
#define ALICEO2_TIMEFRAME_WRITER_DEVICE_H_

#include "O2Device/O2Device.h"
#include <fstream>

namespace o2
{
namespace data_flow
{

/// A device which writes to file the timeframes.
class TimeframeWriterDevice : public base::O2Device
{
 public:
  static constexpr const char* OptionKeyInputChannelName = "input-channel-name";
  static constexpr const char* OptionKeyOutputFileName = "output-file";
  static constexpr const char* OptionKeyMaxTimeframesPerFile = "max-timeframes-per-file";
  static constexpr const char* OptionKeyMaxFileSize = "max-file-size";
  static constexpr const char* OptionKeyMaxFiles = "max-files";

  /// Default constructor
  TimeframeWriterDevice();

  /// Default destructor
  ~TimeframeWriterDevice() override = default;

  void InitTask() final;

  /// The PostRun will trigger saving the file to disk
  void PostRun() final;

 protected:
  /// Overloads the Run() method of FairMQDevice
  void Run() final;

  std::string mInChannelName;
  std::string mOutFileName;
  std::fstream mFile;
  size_t mMaxTimeframes;
  size_t mMaxFileSize;
  size_t mMaxFiles;
  size_t mFileCount;
};

} // namespace data_flow
} // namespace o2

#endif
