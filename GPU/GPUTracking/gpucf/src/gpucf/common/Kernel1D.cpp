#include "Kernel1D.h"

#include <gpucf/common/log.h>


using namespace gpucf;


Kernel1D::Kernel1D(const std::string &name, cl::Program prg)
    : kernel(prg, name.c_str())
    , name(name)
{
}

void Kernel1D::call(
        size_t offset, 
        size_t workitems, 
        size_t local, 
        cl::CommandQueue queue)
{
    ASSERT(workitems > 0);

    try
    {
        queue.enqueueNDRangeKernel(
            kernel,
            cl::NDRange(offset),
            cl::NDRange(workitems),
            cl::NDRange(local),
            nullptr,
            event.get());
    }
    catch (const cl::Error &err)
    {
        log::Error() << "Kernel " << name << " throws error " 
            << err.what() << "(" << log::clErrToStr(err.err()) << ")";

        throw err;
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
