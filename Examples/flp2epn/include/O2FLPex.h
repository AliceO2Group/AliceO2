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

class O2FLPex : public FairMQDevice
{
  public:
    enum {
      NumContent = FairMQDevice::Last,
      Last

    };

    O2FLPex();

    virtual ~O2FLPex();

    virtual void SetProperty(const int key, const std::string &value);

    virtual std::string GetProperty(const int key, const std::string &default_ = "");

    virtual void SetProperty(const int key, const int value);

    virtual int GetProperty(const int key, const int default_ = 0);

  protected:
    int fNumContent;

    virtual void Run();
};

#endif
