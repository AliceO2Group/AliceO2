#pragma once

#include <gpucf/ClEnv.h>
#include <gpucf/common/DataSet.h>
#include <gpucf/common/Measurements.h>

#include <string>


namespace gpucf
{

class GPUAlgorithm
{
public:
    struct Result
    {
        DataSet result;
        std::vector<Measurement> profiling; 
    };

    GPUAlgorithm(const std::string &);

    void setup(ClEnv &, const DataSet &);

    Result run();

    std::string getName() const;
    
protected:
    virtual void setupImpl(ClEnv &, const DataSet &) = 0;
    virtual Result runImpl() = 0;

private:
    bool isSetup = false;

    std::string name;
};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
