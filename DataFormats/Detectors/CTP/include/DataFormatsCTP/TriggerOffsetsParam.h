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

#ifndef _ALICEO2_CTP_TRIGGER_OFFSETS_PARAM_H_
#define _ALICEO2_CTP_TRIGGER_OFFSETS_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

/// \brief Configurable param for CTP trigger offsets (in BCs)

namespace o2
{
namespace ctp
{
struct TriggerOffsetsParam : public o2::conf::ConfigurableParamHelper<TriggerOffsetsParam> {
  static constexpr int MaxNDet = 32; // take with margin to account for possible changes / upgrades
  int64_t LM_L0 = 15;
  int64_t L0_L1 = 280;
  int64_t customOffset[MaxNDet] = {};
  O2ParamDef(TriggerOffsetsParam, "TriggerOffsetsParam"); // boilerplate stuff + make principal key
};
} // namespace ctp

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::ctp::TriggerOffsetsParam> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif // _ALICEO2_CTP_TRIGGER_OFFSETS_RARAM_H_
