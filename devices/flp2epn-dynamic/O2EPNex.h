/**
 * O2EPNex.h
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#ifndef O2EPNEX_H_
#define O2EPNEX_H_

#include <string>

#include "FairMQDevice.h"

struct Content {
  int id;
  double a;
  double b;
  int x;
  int y;
  int z;
};

class O2EPNex: public FairMQDevice
{
  public:
    enum {
      HeartbeatIntervalInMs = FairMQDevice::Last,
      Last
    };
    O2EPNex();
    virtual ~O2EPNex();

    virtual void SetProperty(const int key, const std::string& value);
    virtual std::string GetProperty(const int key, const std::string& default_ = "");
    virtual void SetProperty(const int key, const int value);
    virtual int GetProperty(const int key, const int default_ = 0);

  protected:
    int fHeartbeatIntervalInMs;

    virtual void Run();
};

#endif
