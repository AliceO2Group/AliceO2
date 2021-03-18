//
// Created by rl on 3/17/21.
//

#ifndef STEER_DIGITIZERWORKFLOW_CTPDIGITIZER_H_
#define STEER_DIGITIZERWORKFLOW_CTPDIGITIZER_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace ctp
{

o2::framework::DataProcessorSpec getCTPDigitizerSpec(int channel, bool mctruth = true);

} // namespace ctp
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_CTPDIGITIZER_H_ */
