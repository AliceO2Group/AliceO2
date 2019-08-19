#include "Experiment.h"

#include <gpucf/common/log.h>

#include <fstream>


using namespace gpucf;
namespace fs = filesystem;


Experiment::Experiment(ClusterFinderConfig cfg)
    : cfg(cfg)
{
}

Experiment::~Experiment()
{
}

void Experiment::save(fs::path fname, const Measurements &data)
{
    log::Info() << "Writing measurements to " << fname;

    std::ofstream out(fname.str());
    out << data;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
