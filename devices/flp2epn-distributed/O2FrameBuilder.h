/**
 * O2FrameBuilder.h
 *
 * @since 2014-10-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany
 */

#ifndef O2FRAMEBUILDER_H_
#define O2FRAMEBUILDER_H_

#include "FairMQDevice.h"

using namespace std;

class O2FrameBuilder : public FairMQDevice
{
  public:

    O2FrameBuilder();

    virtual ~O2FrameBuilder();

  protected:
    virtual void Run();
};

#endif /* O2FRAMEBUILDER_H_ */
