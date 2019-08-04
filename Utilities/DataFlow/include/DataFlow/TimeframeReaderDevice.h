// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TIMEFRAME_READER_H_
#define ALICEO2_TIMEFRAME_READER_H_

#include "O2Device/O2Device.h"
#include <fstream>

namespace o2
{
namespace data_flow
{

/// A device which writes to file the timeframes.
class TimeframeReaderDevice : public base::O2Device
{
 public:
  static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";
  static constexpr const char* OptionKeyInputFileName = "input-file";

  /// Default constructor
  TimeframeReaderDevice();

  /// Default destructor
  ~TimeframeReaderDevice() override = default;

  void InitTask() final;

 protected:
  /// Overloads the ConditionalRun() method of FairMQDevice
  bool ConditionalRun() final;

  std::string mOutChannelName;
  std::string mInFileName;
  std::fstream mFile;
  std::vector<std::string> mSeen;
};

} // namespace data_flow
} // namespace o2

#endif
