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
#include "PayloadEncoderImpl.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include "UserLogicElinkEncoder.h"
#include "UserLogicElinkEncoderMerger.h"
#include <gsl/span>

namespace o2::mch::raw
{
namespace impl
{
// cannot partially specialize a function, so create a struct (which can
// be specialized) and use it within the function below.
template <typename FORMAT, typename CHARGESUM, bool forceNoPhase>
struct PayloadEncoderCreator {
  static std::unique_ptr<PayloadEncoder> _(Solar2FeeLinkMapper solar2feelink)
  {
    GBTEncoder<FORMAT, CHARGESUM>::forceNoPhase = forceNoPhase;
    return std::make_unique<PayloadEncoderImpl<FORMAT, CHARGESUM>>(solar2feelink);
  }
};
} // namespace impl

template <typename FORMAT, typename CHARGESUM, bool forceNoPhase>
std::unique_ptr<PayloadEncoder> createPayloadEncoder(Solar2FeeLinkMapper solar2feelink)
{
  return impl::PayloadEncoderCreator<FORMAT, CHARGESUM, forceNoPhase>::_(solar2feelink);
}
std::unique_ptr<PayloadEncoder> createPayloadEncoder(Solar2FeeLinkMapper);

// define only the specializations we use

template std::unique_ptr<PayloadEncoder> createPayloadEncoder<BareFormat, SampleMode, true>(Solar2FeeLinkMapper);
template std::unique_ptr<PayloadEncoder> createPayloadEncoder<BareFormat, SampleMode, false>(Solar2FeeLinkMapper);

template std::unique_ptr<PayloadEncoder> createPayloadEncoder<BareFormat, ChargeSumMode, true>(Solar2FeeLinkMapper);
template std::unique_ptr<PayloadEncoder> createPayloadEncoder<BareFormat, ChargeSumMode, false>(Solar2FeeLinkMapper);

template std::unique_ptr<PayloadEncoder> createPayloadEncoder<UserLogicFormat, SampleMode, true>(Solar2FeeLinkMapper);
template std::unique_ptr<PayloadEncoder> createPayloadEncoder<UserLogicFormat, SampleMode, false>(Solar2FeeLinkMapper);

template std::unique_ptr<PayloadEncoder> createPayloadEncoder<UserLogicFormat, ChargeSumMode, true>(Solar2FeeLinkMapper);
template std::unique_ptr<PayloadEncoder> createPayloadEncoder<UserLogicFormat, ChargeSumMode, false>(Solar2FeeLinkMapper);

} // namespace o2::mch::raw
