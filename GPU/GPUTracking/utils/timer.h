// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file timer.h
/// \author David Rohr

#ifndef QONMODULE_TIMER_H
#define QONMODULE_TIMER_H

class HighResTimer
{
 public:
  HighResTimer() = default;
  ~HighResTimer() = default;
  void Start();
  void Stop();
  void Abort();
  void Reset();
  void ResetStart();
  double GetElapsedTime();
  double GetCurrentElapsedTime(bool reset = false);
  void StopAndStart(HighResTimer& startTimer);
  int IsRunning() { return running; }
  void AddTime(double t);

 private:
  double ElapsedTime = 0.;
  double StartTime = 0.;
  int running = 0;

  static double GetFrequency();
  static double GetTime();
#ifndef GPUCODE
  static double Frequency;
#endif
};

#endif
