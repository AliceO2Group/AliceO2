#pragma once

#include <gpucf/ClEnv.h>
#include <gpucf/common/DataSet.h>
#include <gpucf/common/Measurements.h>


namespace gpucf
{

class GPUAlgorithm
{
public:
    struct Result
    {
        DataSet result;
        Measurements profiling; 
    };

    void setup(ClEnv &, const DataSet &);

    Result run();
    
protected:
    virtual void setupImpl(ClEnv &, const DataSet &) = 0;
    virtual Result runImpl() = 0;

private:
    bool isSetup = false;
 
};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
