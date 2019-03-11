#pragma once

#include <gpucf/common/Timestamp.h>

#include <nonstd/optional.hpp>

#include <CL/cl2.hpp>


namespace gpucf
{

class Event
{

public:
    Event();

    cl::Event *get();

    Timestamp queued()    const;
    Timestamp submitted() const;
    Timestamp start()     const;
    Timestamp end()       const;

private:
    nonstd::optional<cl::Event> event;

    Timestamp profilingInfo(cl_profiling_info) const;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
