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
  int IsRunning() { return running; }

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
