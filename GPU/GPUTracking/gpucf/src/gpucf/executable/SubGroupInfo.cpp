#include "SubGroupInfo.h"

#include <gpucf/common/log.h>


using namespace gpucf;


SubGroupInfo::SubGroupInfo()
    : Executable("Print information about subgroups for given Workgroup size.")
{
}

void SubGroupInfo::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags = std::make_unique<ClEnv::Flags>(required, optional);
    workGroupSizeFlag = OptIntFlag(
            new IntFlag(required, "N", "Workgroup size.", {'w', "workgroup"}));
}

int SubGroupInfo::mainImpl()
{
    ClEnv env(*envFlags); 

    cl::Context context = env.getContext();

    // Load kernels
    cl::Program cfprg = env.buildFromSrc("clusterFinder.cl", {});
    cl::Kernel findClusters(cfprg, "findClusters");

    cl_int err;
    constexpr size_t numDim = 1;
    size_t workGroupSize = workGroupSizeFlag->Get(); 

    size_t subGroupSize;
    err = clGetKernelSubGroupInfoKHR(findClusters.get(),
                                     env.getDevice().get(),
                                     CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
                                     numDim,
                                     &workGroupSize,
                                     numDim,
                                     &subGroupSize,
                                     NULL);
    ASSERT(err == CL_SUCCESS);
    log::Info() << "subgroup size:  " << subGroupSize;


    size_t subGroupCount;
    err = clGetKernelSubGroupInfoKHR(findClusters.get(),
                                     env.getDevice().get(),
                                     CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR,
                                     numDim,
                                     &workGroupSize,
                                     numDim,
                                     &subGroupCount,
                                     NULL);
    ASSERT(err == CL_SUCCESS);
    log::Info() << "subgroup count: " << subGroupSize;

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
