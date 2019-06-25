#include "Experiment.h"

#include <gpucf/common/log.h>

#include <fstream>


using namespace gpucf;
namespace fs = filesystem;


Experiment::Experiment(fs::path base, ClusterFinderConfig cfg)
    : cfg(cfg)
    , baseDir(base)
{
}

Experiment::~Experiment()
{
}

void Experiment::save(fs::path fname, const Measurements &data)
{
    fs::path fullname = baseDir / fname;

    log::Info() << "Writing measurements to " << fullname;

    std::ofstream out(fullname.str());
    out << data;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
