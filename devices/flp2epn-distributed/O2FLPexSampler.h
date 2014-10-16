/**
 * O2FLPexSampler.h
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#ifndef O2FLPEXSAMPLER_H_
#define O2FLPEXSAMPLER_H_

#include <string>

#include "FairMQDevice.h"

using namespace std;

class O2FLPexSampler : public FairMQDevice
{
  public:
    enum {
      EventRate = FairMQDevice::Last,
      EventSize,
      Last
    };

    O2FLPexSampler();
    virtual ~O2FLPexSampler();

    void ResetEventCounter();
    virtual void SetProperty(const int key, const string& value, const int slot = 0);
    virtual string GetProperty(const int key, const string& default_ = "", const int slot = 0);
    virtual void SetProperty(const int key, const int value, const int slot = 0);
    virtual int GetProperty(const int key, const int default_ = 0, const int slot = 0);

  protected:
    virtual void Run();

    int fEventSize;
    int fEventRate;
    int fEventCounter;
};

#endif /* O2FLPEXSAMPLER_H_ */
