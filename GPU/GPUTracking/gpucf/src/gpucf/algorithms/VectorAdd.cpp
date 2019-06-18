#include "VectorAdd.h"

#include <gpucf/common/ClEnv.h>
#include <gpucf/common/log.h>


using namespace gpucf;

namespace fs = filesystem;


VectorAdd::VectorAdd() 
{
    ASSERT(a.size() == N);
    ASSERT(b.size() == N);
    ASSERT(c.size() == N);

    for (size_t i = 0; i < N; i++) 
    {
        a[i] = i;
    }

    b = a;
}

bool VectorAdd::run(ClEnv &env) 
{

    cl::Context context = env.getContext();
    cl::Device device = env.getDevice();

    cl::CommandQueue queue = cl::CommandQueue(context, device);

    cl::Program addPrg = env.buildFromSrc("vecadd.cl");
    cl::Kernel addKernel(addPrg, "vecadd");

    size_t datasize = sizeof(int) * a.size();

    cl::Buffer bufA = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
    cl::Buffer bufB = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
    cl::Buffer bufC = cl::Buffer(context, CL_MEM_WRITE_ONLY, datasize);

    queue.enqueueWriteBuffer(bufA, CL_FALSE, 0, datasize, a.data());
    queue.enqueueWriteBuffer(bufB, CL_FALSE, 0, datasize, b.data());

    addKernel.setArg(0, bufA);
    addKernel.setArg(1, bufB);
    addKernel.setArg(2, bufC);

    cl::NDRange global(N);
    cl::NDRange local(256);
    queue.enqueueNDRangeKernel(addKernel, cl::NullRange, global, local);

    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, datasize, c.data());

    for (size_t i = 0; i < N; i++) 
    {
        if (c[i] != a[i] + b[i]) 
        {
            return false;
        }
    }

    return true;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
