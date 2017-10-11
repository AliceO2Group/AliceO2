// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MOCKUP_STFHANDLER_DEVICE_H_
#define ALICEO2_MOCKUP_STFHANDLER_DEVICE_H_

#include "Common/SubTimeFrameDataModel.h"

#include "O2Device/O2Device.h"

#include <deque>
#include <mutex>
#include <condition_variable>

namespace o2 {
namespace DataDistribution {

class StfHandlerDevice : public Base::O2Device {
public:
  static constexpr const char* OptionKeyInputChannelName = "input-channel-name";
  static constexpr const char* OptionKeyFreeShmChannelName = "free-shm-channel-name";

  /// Default constructor
  StfHandlerDevice();

  /// Default destructor
  ~StfHandlerDevice() override;

  void InitTask() final;

protected:
  bool ConditionalRun() final;

  std::string mInputChannelName;
  std::string mFreeShmChannelName;

  std::uint64_t mDelayUs = 1000;

  std::vector<FairMQMessagePtr> mMessages;
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_MOCKUP_STFHANDLER_DEVICE_H_ */
