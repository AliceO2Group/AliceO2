/**
 * O2EPNex.h
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#ifndef O2EPNEX_H_
#define O2EPNEX_H_

#include "FairMQDevice.h"

struct Content {
  double a;
  double b;
  int x;
  int y;
  int z;
};

class O2EPNex : public FairMQDevice
{
  public:
    O2EPNex();
    virtual ~O2EPNex();
  protected:
    virtual void Run();
};

#endif
