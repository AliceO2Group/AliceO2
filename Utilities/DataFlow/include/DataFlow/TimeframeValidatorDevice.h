// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TIMEFRAMEVALIDATOR_H_
#define ALICEO2_TIMEFRAMEVALIDATOR_H_

#include "O2Device/O2Device.h"

namespace o2
{
namespace data_flow
{

/// A validating device for time frame data (coming from EPN)
class TimeframeValidatorDevice : public base::O2Device
{
 public:
  static constexpr const char* OptionKeyInputChannelName = "input-channel-name";

  /// Default constructor
  TimeframeValidatorDevice();

  /// Default destructor
  ~TimeframeValidatorDevice() override = default;

  void InitTask() final;

 protected:
  /// Overloads the Run() method of FairMQDevice
  void Run() final;

  std::string mInChannelName;
};

} // namespace data_flow
} // namespace o2

#endif
