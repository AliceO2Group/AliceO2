/**
 * O2Merger.h
 *
 * @since 2012-12-06
 * @author D. Klein, A. Rybalchenko, M. Al-Turany
 */

#ifndef O2Merger_H_
#define O2Merger_H_

#include "FairMQDevice.h"


class O2Merger: public FairMQDevice
{
  public:
    O2Merger();
    virtual ~O2Merger();

  protected:
    virtual void Run();
};

#endif /* O2Merger_H_ */
