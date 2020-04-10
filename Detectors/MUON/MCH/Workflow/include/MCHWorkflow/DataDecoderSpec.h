#ifndef O2_MCH_DATADECODERSPEC_H_
#define O2_MCH_DATADECODERSPEC_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mch
{
namespace raw
{

using namespace o2;
using namespace o2::framework;


o2::framework::DataProcessorSpec getDecodingSpec();

} // end namespace raw
} // end namespace mch
} // end namespace o2

#endif

