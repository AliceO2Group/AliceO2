#ifndef QONMODULE_TIMER_H
#define QONMODULE_TIMER_H

class HighResTimer {

public:
	HighResTimer();
	~HighResTimer();
	void Start();
	void Stop();
	void Reset();
	void ResetStart();
	double GetElapsedTime();
	double GetCurrentElapsedTime(bool reset = false);
	int IsRunning() {return running;}

private:
	double ElapsedTime;
	double StartTime;
	int running;

	static double GetFrequency();
	static double GetTime();
#ifndef GPUCODE
	static double Frequency;
#endif
}; 

#endif
