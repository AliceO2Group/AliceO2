#include "Kernel1D.h"


Kernel1D::Kernel1D(const std::string &name, cl::Program prg)
    : name(name)
    , kernel(prg, name)
{
}

void Kernel1D::call(
        size_t workitems, 
        size_t offset, 
        size_t local, 
        cl::CommandQueue queue)
{
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NDRange(offset),
        cl::NDRange(workitems),
        cl::NDRange(local)
        nullptr,
        event.get());
}

// vim: set ts=4 sw=4 sts=4 expandtab:
