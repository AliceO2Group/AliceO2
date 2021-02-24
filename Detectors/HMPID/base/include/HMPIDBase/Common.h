// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   Common.h
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 15/02/2021

#ifndef DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_COMMON_H_
#define DETECTORS_HMPID_BASE_INCLUDE_HMPIDBASE_COMMON_H_

#include <TStopwatch.h>
#include "Framework/Logger.h"

namespace o2
{
namespace hmpid
{

// ------- Execution time functions -------

/// \class ExecutionTimer
/// \brief HMPID Derived class for the Time of Workflows
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

  ~ExecutionTimer() = default;

  /// getElapseLogTime : returns the seconds for the elapsed log message
  /// @return : the number of seconds for the elapsed logging
  Double_t getElapseLogTime() { return mElapseLogTime; };

  /// setElapseLogTime : set the interval for the elapsed logging
  /// @param[in] interval : the seconds of interval for elapsed logging
  void setElapseLogTime(Double_t interval)
  {
    mElapseLogTime = interval;
    return;
  };

  /// start : starts the timer
  void start()
  {
    mStartTime = mTimer.CpuTime();
    mLastLogTime = mStartTime;
    mTimer.Start(false);
    return;
  };

  /// stop : stops the timer
  void stop()
  {
    mTimer.Stop();
    return;
  };

  /// logMes : Out a message on the LOG(INFO) with extra execution time info
  /// @param[in] message : the message to print
  void logMes(std::string const message)
  {
    LOG(INFO) << message << " Execution time = " << (mTimer.CpuTime() - mStartTime);
    mTimer.Continue();
    return;
  };

  /// elapseMes : Out a message on the LOG(INFO) with extra execution time info
  ///             is the set interval was reached
  /// @param[in] message : the message to print
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
