#include "ChargeMap.h"

#include <gpucf/log.h>

#include <shared/tpc.h>


using namespace gpucf;


ChargeMap::ChargeMap(cl::Context context, cl::Program prg, size_t numOfRows)
{
    digitsToChargeMap = std::make_unique<cl::Kernel>(prg, "digitsToChargeMap"); 

    log::Info() << "Found " << numOfRows << " rows";

    const size_t chargeMapSize  = 
        numOfRows * TPC_PADS_PER_ROW_PADDED * TPC_MAX_TIME_PADDED;
    chargeMapBytes =  sizeof(cl_float) * chargeMapSize;

    chargeMap = std::make_unique<cl::Buffer>(
            context, CL_MEM_READ_WRITE, chargeMapBytes);
}

void ChargeMap::enqueueFill(
        cl::CommandQueue queue, 
        cl::Buffer       digits, 
        cl::NDRange      global,
        cl::NDRange      local)
{
    queue.enqueueFillBuffer(*chargeMap, 0.0f, 0, chargeMapBytes);

    digitsToChargeMap->setArg(0, digits);
    digitsToChargeMap->setArg(1, *chargeMap);

    queue.enqueueNDRangeKernel(*digitsToChargeMap, cl::NullRange, global, local);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
