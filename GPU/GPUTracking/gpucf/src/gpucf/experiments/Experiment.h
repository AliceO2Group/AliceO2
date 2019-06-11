#pragma once

#include <gpucf/common/ClEnv.h>
#include <gpucf/common/Measurements.h>

#include <filesystem/path.h>


namespace gpucf
{

class Experiment
{
public:
    Experiment(filesystem::path);

    virtual ~Experiment();

    virtual void run(ClEnv &) = 0;

protected:
    void save(filesystem::path, const Measurements &);

private:
    filesystem::path baseDir;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
