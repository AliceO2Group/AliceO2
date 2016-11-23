#ifndef SCHED_AFFINITY_WIN32_WRAPPER_H
#define SCHED_AFFINITY_WIN32_WRAPPER_H

typedef __int64 cpu_set_t;
typedef HANDLE pid_t;

static inline int CPU_ISSET(__int64 cpu, cpu_set_t *set)
{
	return((*set & ((__int64) 1 << cpu)) ? 1 : 0);
}

static inline int sched_setaffinity(pid_t pid, unsigned int cpusetsize, cpu_set_t *mask)
{
	return(0);
}
static inline int sched_getaffinity(pid_t pid, unsigned int cpusetsize, cpu_set_t *mask)
{
	return(0);
}
static inline void CPU_CLR(__int64 cpu, cpu_set_t *set)
{
	*set &= (~((__int64) 1 << cpu));
}

static inline void CPU_SET(__int64 cpu, cpu_set_t *set)
{
	*set |= ((__int64) 1 << cpu);
}

static inline void CPU_ZERO(cpu_set_t *set)
{
	*set = 0;
}

#endif