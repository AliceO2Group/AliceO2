/**
 * O2FLPex.h
 *
 * @since 2014-02-24
 * @author A. Rybalchenko
 */

#ifndef O2FLPEX_H_
#define O2FLPEX_H_

#include <FairMQDevice.h>

class O2FLPex : public FairMQDevice
{
  public:
    O2FLPex();

    ~O2FLPex() override;

  protected:
    int fNumContent;

    void InitTask() override;
    bool ConditionalRun() override;
};

#endif
