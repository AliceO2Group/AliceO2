#pragma once

#include "DataReader.h"
#include "ClusterParser.h"

#include <args/args.hxx>


namespace gpucf
{

class ClusterReader : public DataReader<Cluster, ClusterParser>
{

public:
    class Flags
    {

    public:
        args::ValueFlag<std::string> infile;
        args::ValueFlag<int> workers;

        Flags(args::Group &required, args::Group &optional)
            : infile(required, "File", "File of ground-truth clusters.",
                    {'c', "cluster"})
            , workers(optional, "N", 
                "Number of workers that parse the cluster file. (default=4)",
                {'v', "cworkers"}, 4)
        {
        }
        
    };

    ClusterReader(const std::string &file, size_t numWorkers) 
        : DataReader<Cluster, ClusterParser>(file, numWorkers)
    {
    }

    ClusterReader(Flags &flags)
        : ClusterReader(args::get(flags.infile), args::get(flags.workers))
    {
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

// vim: set ts=4 sw=4 sts=4 expandtab:
