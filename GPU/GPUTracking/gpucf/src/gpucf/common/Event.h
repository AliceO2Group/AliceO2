#pragma once

#include <gpucf/common/Timestamp.h>

#include <CL/cl2.hpp>


namespace gpucf
{

class Event
{

public:
    cl::Event *get();

    Timestamp startMs() const;

    Timestamp endMs() const;

private:
    cl::Event event;

    Timestamp profilingInfo(cl_profiling_info) const;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
