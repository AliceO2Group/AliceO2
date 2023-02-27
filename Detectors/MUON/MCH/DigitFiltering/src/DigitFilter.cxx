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

#include "MCHDigitFiltering/DigitFilter.h"

#include "DataFormatsMCH/Digit.h"
#include "Framework/Logger.h"
#include "MCHDigitFiltering/DigitFilterParam.h"
#include "MCHMappingInterface/Segmentation.h"
#include <functional>
#include <gsl/span>
#include <vector>

namespace
{
/** function used to select the signal.
 * might cut some signal, the focus here is to kill as
 * much background as possible.
 */
double signalCut(double* x, const double* p)
{
  double x0 = pow(p[0] / p[2], 1. / p[3]) + p[1];
  if (x[0] < x0) {
    return p[0];
  } else {
    return p[2] * pow(x[0] - p[1], p[3]);
  }
}

/** function used to reject the background.
 * might not kill all background, focus here is ensure
 * we're not killing some signal along the way.
 */
double backgroundCut(double* x, const double* p)
{

  double x0 = (p[3] * p[2] - p[1] * p[0]) / (p[3] - p[1]);
  if (x[0] < x0) {
    return p[1] * (x[0] - p[0]);
  } else {
    return p[3] * (x[0] - p[2]);
  }
}

o2::mch::DigitFilter createMinAdcCut(uint32_t minADC)
{
  return [minADC](const o2::mch::Digit& digit) -> bool {
    if (digit.getADC() < minADC) {
      return false;
    }
    return true;
  };
}

o2::mch::DigitFilter createRejectBackground()
{
  uint16_t minNSamplesBackground = 14;
  double backgroundParam[4] = {18., 24., -20., 7.0};

  auto backgroundCut = [backgroundParam](double* x) {
    return ::backgroundCut(x, backgroundParam);
  };

  return [backgroundCut, minNSamplesBackground](const o2::mch::Digit& digit) -> bool {
    double nSample = digit.getNofSamples();
    if (digit.getNofSamples() < minNSamplesBackground || digit.getADC() < backgroundCut(&nSample)) {
      return false;
    }
    return true;
  };
}

o2::mch::DigitFilter createSelectSignal()
{
  uint16_t minNSamplesSignal = 17;
  double signalParam[4] = {80., 16., 12., 1.2};

  auto signalCut = [signalParam](double* x) {
    return ::signalCut(x, signalParam);
  };
  return [signalCut, minNSamplesSignal](const o2::mch::Digit& digit) -> bool {
    double nSample = digit.getNofSamples();
    if (digit.getNofSamples() < minNSamplesSignal || digit.getADC() < signalCut(&nSample)) {
      return false;
    }
    return true;
  };
}
} // namespace

namespace o2::mch
{
void report(const std::map<int, std::vector<int>>& rejectList, uint32_t statusMask)
{
  int nbad{0};
  for (const auto it : rejectList) {
    nbad += it.second.size();
  }
  LOGP(info, "Got {} bad channels in {} detection element{} (using statusMask=0x{:x})", nbad,
       rejectList.size(), rejectList.size() > 1 ? "s" : "", statusMask);
}

DigitFilter createBadChannelFilter(const StatusMap& statusMap,
                                   uint32_t statusMask)
{
  auto rejectList = applyMask(statusMap, statusMask);
  report(rejectList, statusMask);

  return [rejectList, statusMask](const o2::mch::Digit& digit) -> bool {
    bool goodChannel{true};
    auto deID = digit.getDetID();
    auto it = rejectList.find(digit.getDetID());
    if (it != rejectList.end()) {
      // channel is good if it's not found in the rejectlist
      goodChannel = std::find(it->second.begin(), it->second.end(), digit.getPadID()) == it->second.end();
    }
    return goodChannel;
  };
}

DigitFilter createDigitFilter(uint32_t minADC,
                              bool rejectBackground,
                              bool selectSignal,
                              const StatusMap& statusMap,
                              uint32_t statusMask)
{
  std::vector<DigitFilter> parts;

  if (minADC > 0) {
    parts.emplace_back(createMinAdcCut(minADC));
  }
  if (rejectBackground) {
    parts.emplace_back(createRejectBackground());
  }
  if (selectSignal) {
    parts.emplace_back(createSelectSignal());
  }
  if (!statusMap.empty() && statusMask) {
    parts.emplace_back(createBadChannelFilter(statusMap, statusMask));
  }
  return [parts](const Digit& digit) {
    for (const auto& p : parts) {
      if (!p(digit)) {
        return false;
      }
    }
    return true;
  };
}

} // namespace o2::mch
