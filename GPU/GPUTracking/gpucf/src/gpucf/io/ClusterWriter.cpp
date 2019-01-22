#include "ClusterWriter.h"

#include <gpucf/log.h>

#include <fstream>


using namespace gpucf;


void ClusterWriter::write(const std::vector<Cluster> &clusters)
{
    log::Info() << "Writing clusters to file " << fName << ".";
    std::ofstream out(fName);

    for (const Cluster &c : clusters)
    {
        out << serialize(c) << "\n"; 
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:

