#pragma once

#include <CL/cl2.hpp>


namespace gpucf
{
    
class Event
{

public:
    cl::Event *get();

    float startMs() const;

    float endMs() const;

private:
    static float nsToMs(cl_ulong);

    cl::Event event;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
