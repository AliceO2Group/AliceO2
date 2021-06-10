// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "BareElinkEncoder.h"
#include "BareElinkEncoderMerger.h"
#include "MCHRawCommon/DataFormats.h"
#include "PayloadEncoderImpl.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "UserLogicElinkEncoder.h"
#include "UserLogicElinkEncoderMerger.h"
#include <gsl/span>

namespace o2::mch::raw
{
namespace impl
{
template <typename FORMAT, typename CHARGESUM, int VERSION, bool forceNoPhase>
struct PayloadEncoderCreator {
  static std::unique_ptr<PayloadEncoder> _(Solar2FeeLinkMapper solar2feelink)
  {
    GBTEncoder<FORMAT, CHARGESUM, VERSION>::forceNoPhase = forceNoPhase;
    return std::make_unique<PayloadEncoderImpl<FORMAT, CHARGESUM, VERSION>>(solar2feelink);
  }
};
} // namespace impl

template <typename FORMAT, typename CHARGESUM, int VERSION, bool forceNoPhase = true>
std::unique_ptr<PayloadEncoder> createPayloadEncoder(Solar2FeeLinkMapper solar2feelink)
{
  return impl::PayloadEncoderCreator<FORMAT, CHARGESUM, VERSION, forceNoPhase>::_(solar2feelink);
}

std::unique_ptr<PayloadEncoder> createPayloadEncoder(Solar2FeeLinkMapper solar2feelink,
                                                     bool userLogic, int version, bool chargeSumMode)
{
  if (version != 0 && version != 1) {
    throw std::invalid_argument("Only version 0 or 1 are supported");
  }
  if (userLogic) {
    if (version == 0) {
      if (chargeSumMode) {
        return createPayloadEncoder<UserLogicFormat, ChargeSumMode, 0>(solar2feelink);
      } else {
        return createPayloadEncoder<UserLogicFormat, SampleMode, 0>(solar2feelink);
      }
    } else {
      if (chargeSumMode) {
        return createPayloadEncoder<UserLogicFormat, ChargeSumMode, 1>(solar2feelink);
      } else {
        return createPayloadEncoder<UserLogicFormat, SampleMode, 1>(solar2feelink);
      }
    }
  } else {
    if (chargeSumMode) {
      return createPayloadEncoder<BareFormat, ChargeSumMode, 0>(solar2feelink);
    } else {
      return createPayloadEncoder<BareFormat, SampleMode, 0>(solar2feelink);
    }
  }
}
} // namespace o2::mch::raw
