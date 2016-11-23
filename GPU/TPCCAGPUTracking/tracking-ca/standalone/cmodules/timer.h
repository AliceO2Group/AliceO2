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
	double GetCurrentElapsedTime();

private:
	static double Frequency;
	static double GetFrequency();

	double ElapsedTime;
	double StartTime;
	int running;
}; 

#endif
