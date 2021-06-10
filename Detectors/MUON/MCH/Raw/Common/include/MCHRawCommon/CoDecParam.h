#ifndef O2_MCH_RAW_CODEC_PARAM_H
#define O2_MCH_RAW_CODEC_PARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

struct CoDecParam : public o2::conf::ConfigurableParamHelper<CoDecParam> {

  int sampaBcOffset = 339986; // default global sampa bunch-crossing offset

  O2ParamDef(CoDecParam, "MCHCoDecParam")
};

} // namespace o2::mch

#endif
