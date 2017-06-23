// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATARELAYER_H
#define FRAMEWORK_DATARELAYER_H

#include <fairmq/FairMQMessage.h>
#include "Framework/InputSpec.h"
#include <cstddef>
#include <map>
#include <vector>

namespace o2 {
namespace framework {

class DataRelayer {
public:
  enum RelayChoice {
    WillRelay,
    WillNotRelay
  };

  using InputsMap = std::map<std::string, InputSpec>;
  using ForwardsMap = std::map<std::string, InputSpec>;

  struct TimeframeId {
    size_t value;
  };

  struct PartRef {
    TimeframeId timeframeId;
    size_t partPos;
    std::unique_ptr<FairMQMessage> header;
    std::unique_ptr<FairMQMessage> payload;
    bool operator<(const PartRef& rhs) const {
      return std::tie(timeframeId.value, partPos) < std::tie(rhs.timeframeId.value, rhs.partPos);
    }
  };

  /// This is used to communicate the parts which are ready to be processed and
  /// those which are ready to be forwaded.
  /// readyInputs is a vector of parts which can be be processed, sorted by timeframe and
  /// then position in the argument bindings.
  struct DataReadyInfo {
    std::vector<PartRef> readyInputs;
  };

  DataRelayer(const InputsMap &, const ForwardsMap&);

  RelayChoice relay(std::unique_ptr<FairMQMessage> &&header,
                    std::unique_ptr<FairMQMessage> &&payload);

  DataReadyInfo getReadyToProcess();
  // The messages which need to be forwarded to next stage.
  const std::vector<bool> &forwardingMask();
private:
  static constexpr TimeframeId sInvalidTimeframeId{(size_t) -1};
  struct CompletionMask {
    TimeframeId timeframeId;
    size_t mask;
  };
  InputsMap mInputs;
  ForwardsMap mForwards;
  std::vector<PartRef> mCache;
  std::vector<CompletionMask> mCompletion;
  std::vector<bool> mForwardingMask;
  size_t mAllInputsMask;
};

}
}

#endif
