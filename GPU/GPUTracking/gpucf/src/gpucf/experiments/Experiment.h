#pragma once

#include <gpucf/ClEnv.h>
#include <gpucf/CsvFile.h>

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
    void saveFile(filesystem::path, const CsvFile &);

private:
    filesystem::path baseDir;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
