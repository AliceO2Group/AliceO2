/**
 * O2EpnMerger.h
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#ifndef O2EpnMerger_H_
#define O2EpnMerger_H_

#include "FairMQDevice.h"

class O2EpnMerger : public FairMQDevice
{
  public:
    O2EpnMerger();

    virtual ~O2EpnMerger();

  protected:
    virtual void Run();
};

#endif /* O2EpnMerger_H_ */
