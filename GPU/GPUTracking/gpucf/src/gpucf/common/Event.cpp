#include "Event.h"


using namespace gpucf;


cl::Event *Event::get()
{
    return &event;
}

float Event::executionTimeMs() const
{
    cl_ulong start;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);

    cl_ulong end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);

    cl_ulong duration = end - start;

    return duration / float(1000000);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
