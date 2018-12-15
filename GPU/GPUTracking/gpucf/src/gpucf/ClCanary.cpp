#include "ClCanary.h"

#include <gpucf/ClEnv.h>

void ClCanary::run(ClEnv &env) {

    cl::Context context = env.getContext();
    cl::Device device = env.getDevice();

    cl::CommandQueue queue = cl::CommandQueue(context, device);

    cl::Program::Sources source = env.loadSrc("vector_add.cl");

    cl::Buffer bufA = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
    cl::Buffer bufB = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
    cl::Buffer bufC = cl::Buffer(context, CL_MEM_WRITE_ONLY, datasize);


     
}

// vim: set ts=4 sw=4 sts=4 expandtab:
