#include "TimeCf.h"

#include <gpucf/common/log.h>


using namespace gpucf;
namespace fs = filesystem;


TimeCf::TimeCf(
        const std::string &n,
        fs::path tgt,
        ClusterFinderConfig conf,
        nonstd::span<const Digit> d, 
        size_t N, 
        fs::path baseDir)
    : Experiment(baseDir)
    , name(n)
    , tgtFile(tgt)
    , config(conf)
    , repeats(N)
    , digits(d)
{
}

void TimeCf::run(ClEnv &env)
{

    log::Info() << "Benchmarking " << name;

    Measurements measurements;

    for (size_t i = 0; i < repeats+1; i++)
    {
        ClEnv envCopy = env;
        GPUClusterFinder cf; 
        cf.setup(config, envCopy, digits);

        auto res = cf.run();

        if (i > 0)
        {
            measurements.add(res.profiling);
            measurements.finishRun();
        }
    }

    save(tgtFile, measurements);
}
    

// vim: set ts=4 sw=4 sts=4 expandtab:
