#pragma once

#include "DataReader.h"
#include "ClusterParser.h"

#include <args/args.hxx>


namespace gpucf
{

using ClusterReader = DataReader<Cluster, ClusterParser>;

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
