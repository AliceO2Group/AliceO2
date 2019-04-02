#ifndef ALICEO2_FIT_HITTYPE_H_
#define ALICEO2_FIT_HITTYPE_H_

#include "SimulationDataFormat/BaseHits.h"
#include "SimulationDataFormat/Stack.h"
#include "CommonUtils/ShmAllocator.h"

namespace o2
{
namespace fit
{
class HitType : public o2::BasicXYZEHit<float>
{
 public:
  using BasicXYZEHit<float>::BasicXYZEHit;
};

} // namespace fit

} // namespace o2

#endif

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::fit::HitType> : public o2::utils::ShmAllocator<o2::fit::HitType>
{
};
} // namespace std
#endif
