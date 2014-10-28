/**
 * FrameBuilder.h
 *
 * @since 2014-10-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany
 */

#ifndef ALICEO2_DEVICES_FRAMEBUILDER_H_
#define ALICEO2_DEVICES_FRAMEBUILDER_H_

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

class FrameBuilder : public FairMQDevice
{
  public:
    FrameBuilder();
    virtual ~FrameBuilder();

  protected:
    virtual void Run();
};

} // namespace Devices
} // namespace AliceO2

#endif
