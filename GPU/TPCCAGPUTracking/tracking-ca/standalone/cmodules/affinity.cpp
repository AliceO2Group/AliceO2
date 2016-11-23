#ifdef _WIN32
#include "pthread_mutex_win32_wrapper.h"
#else
#include <sys/types.h>
#include <sys/syscall.h>
#include <syscall.h>
#include <dirent.h>
#include <pthread.h>
#endif
#include <vector>
#include "affinity.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "os_low_level_helper.h"

#ifndef STD_OUT
#define STD_OUT stdout
#endif

pid_t gettid()
{
#ifdef _WIN32
	return((pid_t) GetCurrentThreadId());
#else
	return((pid_t) syscall(SYS_gettid));
#endif
}

#ifdef _WIN32
pid_t getpid()
{
	return((pid_t) GetCurrentProcessId());
}
#endif

struct threadNameStruct
{
	pid_t thread_id;
	std::string name;
};

class lockClass
{
public:
    lockClass() {pthread_mutex_init(&lock, NULL);}
    ~lockClass() {pthread_mutex_destroy(&lock);}
    std::vector<threadNameStruct> threadNames;
    pthread_mutex_t lock;
};

static lockClass lockedVector;

void setThreadName(char* name)
{
	threadNameStruct tmp;
	tmp.thread_id = gettid();
	tmp.name = name;
	pthread_mutex_lock(&lockedVector.lock);
	lockedVector.threadNames.push_back(tmp);
	pthread_mutex_unlock(&lockedVector.lock);
}

void setUnknownNames(char* name)
{
	pid_t pid = getpid();
#ifndef _WIN32
	char dirname[1024];
	sprintf(dirname, "/proc/%d/task", (int) pid);
	DIR* dp = opendir(dirname);
	if (dp)
	{
		dirent* ent;
		while ((ent = readdir(dp)) != NULL)
		{
			pid_t tid = atoi(ent->d_name);
			if (tid != 0 && tid != pid)
			{
				int found = false;
				for (size_t i = 0;i < lockedVector.threadNames.size();i++)
				{
					if (lockedVector.threadNames[i].thread_id == tid)
					{
						found = true;
					}
				}
				if (found == false)
				{
					threadNameStruct tmp;
					tmp.thread_id = tid;
					tmp.name = name;
					lockedVector.threadNames.push_back(tmp);
				}
			}
		}
	}
#endif
}

void setUnknownAffinity(int count, int* cores)
{
	pid_t pid = getpid();
#ifndef _WIN32
	char dirname[1024];
	sprintf(dirname, "/proc/%d/task", (int) pid);
	DIR* dp = opendir(dirname);
	if (dp)
	{
		dirent* ent;
		while ((ent = readdir(dp)) != NULL)
		{
			pid_t tid = atoi(ent->d_name);
			if (tid != 0 && tid != pid)
			{
				int found = false;
				for (size_t i = 0;i < lockedVector.threadNames.size();i++)
				{
					if (lockedVector.threadNames[i].thread_id == tid)
					{
						found = true;
					}
				}
				if (found == false)
				{
					cpu_set_t tmpset;
					CPU_ZERO(&tmpset);
					for (int i = 0;i < count;i++) CPU_SET(cores[i], &tmpset);
					sched_setaffinity(tid, sizeof(tmpset), &tmpset);
				}
			}
		}
	}
#endif
}

void printThreadPinning()
{
	pid_t pid = getpid();
#ifndef _WIN32
	char dirname[1024];
	sprintf(dirname, "/proc/%d/task", (int) pid);
	DIR* dp = opendir(dirname);
	if (dp)
	{
		dirent* ent;
		fprintf(STD_OUT, "%12s", "");
		for (int i = 0;i < get_number_of_cpu_cores();i++)
		{
			fprintf(STD_OUT, " %2d", i);
		}
		fprintf(STD_OUT, "\n");
		
		while ((ent = readdir(dp)) != NULL)
		{
			pid_t tid = atoi(ent->d_name);
			if (tid != 0)
			{
				fprintf(STD_OUT, "Thread %5d", tid);
				cpu_set_t threadmask;
				sched_getaffinity(tid, sizeof(threadmask), &threadmask);
				for (int i = 0;i < get_number_of_cpu_cores();i++)
				{
					if (CPU_ISSET(i, &threadmask))
					{
						fprintf(STD_OUT, "  X");
					}
					else
					{
						fprintf(STD_OUT, "  .");
					}
				}
				fprintf(STD_OUT, " - ");
				bool found = false;
				for (size_t i = 0;i < lockedVector.threadNames.size();i++)
				{
					if (lockedVector.threadNames[i].thread_id == tid)
					{
						fprintf(STD_OUT, "%s", lockedVector.threadNames[i].name.c_str());
						found = true;
						break;
					}
				}
				if (found == false) fprintf(STD_OUT, "Unknown Thread");
				if (CPU_COUNT(&threadmask) == 1)
				{
					for (int i = 0;i < get_number_of_cpu_cores();i++)
					{
						if (CPU_ISSET(i, &threadmask)) fprintf(STD_OUT, " - Pinned to core %d", i);
					}
				}
				char filename[1024];
				sprintf(filename, "/proc/%d/task/%d/stat", (int) pid, (int) tid);
				FILE* fp = fopen(filename, "r");
				if (fp != NULL)
				{
					char buffer[1024];
					fgets(buffer, 1023, fp);
					int count = 0;
					for (unsigned int i = 0;i < strlen(buffer);i++)
					{
						if (buffer[i] == ' ')
						{
							if (++count == 13)
							{
								int time;
								sscanf(&buffer[i + 1], "%d ", &time);
								fprintf(STD_OUT, " - Time: %d", time);
								break;
							}
						}
					}
					fclose(fp);
				}
				fprintf(STD_OUT, "\n");
			}
		}
		closedir(dp);
	}
#endif
}
