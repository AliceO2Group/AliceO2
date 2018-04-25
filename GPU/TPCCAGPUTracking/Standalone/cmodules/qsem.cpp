#include <errno.h>
#include <stdio.h>

#include "qsem.h"

#ifndef STD_OUT
#define STD_OUT stdout
#endif

qSem::qSem(int num)
{
	max = num;
	if (sem_init(&sem, 0, num)) fprintf(STD_OUT, "Error initializing semaphore");
}

qSem::~qSem()
{
	if (sem_destroy(&sem)) fprintf(STD_OUT, "Error destroying semaphore");
}

int qSem::Lock()
{
	int retVal;
	if ((retVal = sem_wait(&sem))) fprintf(STD_OUT, "Error locking semaphore");
	return(retVal);
}

int qSem::Unlock()
{
	int retVal;
	if ((retVal = sem_post(&sem))) fprintf(STD_OUT, "Error unlocking semaphire");
	return(retVal);
}

int qSem::Trylock()
{
	int retVal = sem_trywait(&sem);
	if (retVal)
	{
		if (errno == EAGAIN) return(EBUSY);
		return(-1);
	}
	return(0);
}

#ifndef _WIN32
int qSem::Query()
{
	int value;
	if (sem_getvalue(&sem, &value) != 0) value = -1;
	return(value);
}
#endif
