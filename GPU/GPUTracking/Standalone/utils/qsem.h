#ifndef QSEM_H
#define QSEM_H

#ifdef _WIN32
#include "pthread_mutex_win32_wrapper.h"
#else
#include <semaphore.h>
#endif

class qSem
{
public:
	qSem(int num = 1);
	~qSem();

	int Lock();
	int Unlock();
	int Trylock();
	int Query();

private:
	int max;
	sem_t sem;
};

class qSignal
{
private:
	qSem sem;

public:
	qSignal() : sem(0) {}
	void Wait() {sem.Lock();}
	void Signal() {sem.Unlock();}
};

#endif
