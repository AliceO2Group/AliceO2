/**
 * O2FLPex.h
 *
 * @since 2014-02-24
 * @author A. Rybalchenko
 */

#ifndef O2FLPEX_H_
#define O2FLPEX_H_

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

class O2FLPex: public FairMQDevice
{
  public:
    enum {
      InputFile = FairMQDevice::Last,
      EventSize,
      Last,
      OutputHeartbeat,
      HeartbeatTimeoutInMs
    };
    O2FLPex();
    virtual ~O2FLPex();
    void Log(int intervalInMs);

    virtual void SetProperty(const int key, const string& value, const int slot = 0);
    virtual string GetProperty(const int key, const string& default_ = "", const int slot = 0);
    virtual void SetProperty(const int key, const int value, const int slot = 0);
    virtual int GetProperty(const int key, const int default_ = 0, const int slot = 0);
    virtual void SetProperty(const int key, const boost::posix_time::ptime value, const int slot = 0);
    virtual boost::posix_time::ptime GetProperty(const int key, const boost::posix_time::ptime value, const int slot = 0);

  protected:
    int fEventSize;

    virtual void Init();
    virtual void Run();
    
    int fHeartbeatTimeoutInMs;
    
  private:
    vector<boost::posix_time::ptime> fOutputHeartbeat;
    bool updateIPHeartbeat (string str);
};

#endif
