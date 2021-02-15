/*
 * Common.h
 *
 *  Created on: 5 feb 2021
 *      Author: fap
 */

#ifndef DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_COMMON_H_
#define DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_COMMON_H_

#include <TStopwatch.h>
#include "Framework/Logger.h"

namespace o2
{
namespace hmpid
{

// ------- Execution time functions
class ExecutionTimer
{
 private:
  TStopwatch mTimer;
  Double_t mStartTime;
  Double_t mLastLogTime;
  Double_t mElapseLogTime;

 public:
  ExecutionTimer()
  {
    mStartTime = 0;
    mLastLogTime = 0;
    mElapseLogTime = 10; // default 10 seconds
  };

  ~ExecutionTimer(){};

  Double_t getElapseLogTime() { return mElapseLogTime; };
  void setElapseLogTime(Double_t interval)
  {
    mElapseLogTime = interval;
    return;
  };

  void start()
  {
    mStartTime = mTimer.CpuTime();
    mLastLogTime = mStartTime;
    mTimer.Start(false);
    return;
  };

  void stop()
  {
    mTimer.Stop();
    return;
  };

  void logMes(std::string const message)
  {
    LOG(INFO) << message << " Execution time = " << (mTimer.CpuTime() - mStartTime);
    mTimer.Continue();
    return;
  };

  void elapseMes(std::string const message)
  {
    if (mTimer.CpuTime() - mLastLogTime > mElapseLogTime) {
      LOG(INFO) << message << " Execution time = " << (mTimer.CpuTime() - mStartTime);
      mLastLogTime = mTimer.CpuTime();
    }
    mTimer.Continue();
    return;
  };
};

} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_COMMON_H_ */
