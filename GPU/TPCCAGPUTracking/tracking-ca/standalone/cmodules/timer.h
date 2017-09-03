#ifndef TIMER_H
#define TIMER_H

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

private:
	double ElapsedTime;
	double StartTime;
	int running;

	static double GetFrequency();
#ifndef GPUCODE
	static double Frequency;
#endif
}; 

#endif
