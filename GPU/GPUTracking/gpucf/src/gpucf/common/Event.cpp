#include "Event.h"


using namespace gpucf;


float Event::nsToMs(cl_ulong ns)
{
    return ns / float(1000000);
}


cl::Event *Event::get()
{
    return &event;
}

float Event::startMs() const
{
    cl_ulong start;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);

    return nsToMs(start);
}

float Event::endMs() const
{
    cl_ulong end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);

    return nsToMs(end);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
