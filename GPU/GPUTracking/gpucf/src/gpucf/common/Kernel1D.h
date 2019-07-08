#pragma once

#include <gpucf/common/Event.h>

#include <CL/cl2.hpp>


namespace gpucf
{

class Kernel1D
{

public:
    Kernel1D(const std::string &, cl::Program);

    std::string getName() const
    { 
        return name;
    }

    const Event &getEvent() const
    {
        return event;
    }

protected:
    cl::Kernel kernel;

    void call(size_t, size_t, size_t, cl::CommandQueue);

private:
    std::string name;
    Event event;
    
};

} // namespace gpucf

#define DECL_KERNEL(type, name) type(cl::Program prg) : Kernel1D(name, prg) {}

// vim: set ts=4 sw=4 sts=4 expandtab:
