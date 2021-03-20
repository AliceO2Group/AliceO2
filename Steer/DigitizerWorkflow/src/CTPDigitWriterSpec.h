//
// Created by rl on 3/19/21.
//

#ifndef STEER_DIGITIZERWORKFLOW_CTPDIGITWRITERSPEC_H
#define STEER_DIGITIZERWORKFLOW_CTPDIGITWRITERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsCTP/Digits.h"

namespace o2
{
namespace ctp
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getCTPDigitWriterSpec(bool mctruth)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("CTPDigitWriter",
                                "CTPdigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::ctp::CTPdigit>>{InputSpec{"digit", "CTP", "DIGITSBC"}, "CTPDIGITSBC"})();
}

} // namespace ctp
} // end namespace o2
#endif //STEER_DIGITIZERWORKFLOW_CTPDIGITWRITERSPEC_H
