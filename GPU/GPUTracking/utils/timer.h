// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
