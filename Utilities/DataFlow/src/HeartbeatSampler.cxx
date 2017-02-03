#include "DataFlow/HeartbeatSampler.h"

AliceO2::DataFlow::HeartbeatSampler::HeartbeatSampler()
  : O2Device()
{
}

AliceO2::DataFlow::HeartbeatSampler::~HeartbeatSampler()
{
}

void AliceO2::DataFlow::HeartbeatSampler::InitTask()
{
}

bool AliceO2::DataFlow::HeartbeatSampler::ConditionalRun()
{
  return true;
}
