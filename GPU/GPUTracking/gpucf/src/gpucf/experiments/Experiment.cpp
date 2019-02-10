#include "Experiment.h"

#include <gpucf/common/log.h>

#include <fstream>


using namespace gpucf;
namespace fs = filesystem;


Experiment::Experiment(fs::path base)
    : baseDir(base)
{
}

Experiment::~Experiment()
{
}

void Experiment::saveFile(fs::path fname, const CsvFile &csv)
{
    fs::path fullname = baseDir / fname;

    log::Info() << "Writing measurements to " << fullname;

    std::ofstream out(fullname.str());
    out << csv.str();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
