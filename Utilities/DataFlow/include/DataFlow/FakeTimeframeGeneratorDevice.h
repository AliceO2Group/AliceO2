// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FAKE_TIMEFRAME_GENERATOR_H_
#define ALICEO2_FAKE_TIMEFRAME_GENERATOR_H_

#include "O2Device/O2Device.h"

namespace o2
{
namespace data_flow
{

/// A device which writes to file the timeframes.
class FakeTimeframeGeneratorDevice : public base::O2Device
{
 public:
  static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";
  static constexpr const char* OptionKeyMaxTimeframes = "max-timeframes";

  /// Default constructor
  FakeTimeframeGeneratorDevice();

  /// Default destructor
  ~FakeTimeframeGeneratorDevice() override = default;

  void InitTask() final;

 protected:
  /// Overloads the ConditionalRun() method of FairMQDevice
  bool ConditionalRun() final;

  std::string mOutChannelName;
  size_t mMaxTimeframes;
  size_t mTimeframeCount;
};

} // namespace data_flow
} // namespace o2

#endif
