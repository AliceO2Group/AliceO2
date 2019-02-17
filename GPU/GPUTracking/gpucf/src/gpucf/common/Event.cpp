#include "Event.h"


using namespace gpucf;


cl::Event *Event::get()
{
    return &event;
}

Timestamp Event::startMs() const
{
    return profilingInfo(CL_PROFILING_COMMAND_START);
}

Timestamp Event::endMs() const
{
    return profilingInfo(CL_PROFILING_COMMAND_END);
}

Timestamp Event::profilingInfo(cl_profiling_info key) const
{
    Timestamp data; 
    event.getProfilingInfo(key, &data);

    return data;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
