/**
 * O2Proxy.h
 *
 * @since 2013-10-02
 * @author A. Rybalchenko, , M.Al-Turany
 */

#ifndef O2Proxy_H_
#define O2Proxy_H_

#include "FairMQDevice.h"

class O2Proxy : public FairMQDevice
{
  public:
    O2Proxy();

    virtual ~O2Proxy();

  protected:
    virtual void Run();
};

#endif /* O2Proxy_H_ */
