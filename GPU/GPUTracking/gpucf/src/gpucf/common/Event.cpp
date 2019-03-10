#include "Event.h"


using namespace gpucf;


Event::Event()
    : event(cl::Event())
{
}

cl::Event *Event::get()
{
    return &event.value();
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
    event->getProfilingInfo(key, &data);

    return data;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
