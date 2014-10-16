/**
 * O2FLPex.h
 *
 * @since 2014-02-24
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#ifndef O2FLPEX_H_
#define O2FLPEX_H_

#include <string>
#include <queue>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "FairMQDevice.h"

using namespace std;

using boost::posix_time::ptime;

class O2FLPex : public FairMQDevice
{
  public:
    enum {
      OutputHeartbeat  = FairMQDevice::Last,
      HeartbeatTimeoutInMs,
      NumFLPs,
      SendOffset,
      Last
    };

    O2FLPex();
    virtual ~O2FLPex();

    virtual void SetProperty(const int key, const string& value, const int slot = 0);
    virtual string GetProperty(const int key, const string& default_ = "", const int slot = 0);
    virtual void SetProperty(const int key, const int value, const int slot = 0);
    virtual int GetProperty(const int key, const int default_ = 0, const int slot = 0);
    virtual void SetProperty(const int key, const ptime value, const int slot = 0);
    virtual ptime GetProperty(const int key, const ptime value, const int slot = 0);

  protected:
    virtual void Init();
    virtual void Run();

  private:
    bool updateIPHeartbeat(string str);

    int fHeartbeatTimeoutInMs;
    int fSendOffset;
    queue<FairMQMessage*> fIdBuffer;
    queue<FairMQMessage*> fDataBuffer;
    vector<ptime> fOutputHeartbeat;
};

#endif
