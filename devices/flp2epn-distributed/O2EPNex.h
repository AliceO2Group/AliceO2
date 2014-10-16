/**
 * O2EPNex.h
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany, C. Kouzinopoulos
 */

#ifndef O2EPNEX_H_
#define O2EPNEX_H_

#include <string>

#include "FairMQDevice.h"

using namespace std;

class O2EPNex : public FairMQDevice
{
  public:
    enum {
      HeartbeatIntervalInMs = FairMQDevice::Last,
      Last
    };
    O2EPNex();
    virtual ~O2EPNex();

    virtual void SetProperty(const int key, const string& value, const int slot = 0);
    virtual string GetProperty(const int key, const string& default_ = "", const int slot = 0);
    virtual void SetProperty(const int key, const int value, const int slot = 0);
    virtual int GetProperty(const int key, const int default_ = 0, const int slot = 0);

  protected:
    virtual void Run();
    void sendHeartbeats();

    int fHeartbeatIntervalInMs;
};

#endif
