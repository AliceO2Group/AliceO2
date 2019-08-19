#pragma once

#include <gpucf/common/ClEnv.h>
#include <gpucf/common/Measurements.h>

#include <filesystem/path.h>


namespace gpucf
{

class Experiment
{
public:
    Experiment(ClusterFinderConfig cfg);

    virtual ~Experiment();

    virtual void run(ClEnv &) = 0;

    ClusterFinderConfig getConfig() const 
    {
        return cfg;
    }

protected:
    ClusterFinderConfig cfg;

    void save(filesystem::path, const Measurements &);

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
