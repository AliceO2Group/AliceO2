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
#include "EncoderImpl.h"
#include "MCHRawEncoder/Encoder.h"
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
struct EncoderCreator {
  static std::unique_ptr<Encoder> _()
  {
    GBTEncoder<FORMAT, CHARGESUM>::forceNoPhase = forceNoPhase;
    return std::make_unique<EncoderImpl<FORMAT, CHARGESUM>>();
  }
};
} // namespace impl

template <typename FORMAT, typename CHARGESUM, bool forceNoPhase>
std::unique_ptr<Encoder> createEncoder()
{
  return impl::EncoderCreator<FORMAT, CHARGESUM, forceNoPhase>::_();
}
std::unique_ptr<Encoder> createEncoder();

// define only the specializations we use

template std::unique_ptr<Encoder> createEncoder<BareFormat, SampleMode, true>();
template std::unique_ptr<Encoder> createEncoder<BareFormat, SampleMode, false>();

template std::unique_ptr<Encoder> createEncoder<BareFormat, ChargeSumMode, true>();
template std::unique_ptr<Encoder> createEncoder<BareFormat, ChargeSumMode, false>();

template std::unique_ptr<Encoder> createEncoder<UserLogicFormat, SampleMode, true>();
template std::unique_ptr<Encoder> createEncoder<UserLogicFormat, SampleMode, false>();

template std::unique_ptr<Encoder> createEncoder<UserLogicFormat, ChargeSumMode, true>();
template std::unique_ptr<Encoder> createEncoder<UserLogicFormat, ChargeSumMode, false>();

} // namespace o2::mch::raw
