#include "TimeCf.h"

#include <gpucf/common/log.h>


using namespace gpucf;
namespace fs = filesystem;


TimeCf::TimeCf(
        const std::string &n,
        fs::path tgt,
        GPUClusterFinder::Config conf,
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
    GPUClusterFinder cf; 

    cf.setup(config, env, digits);

    log::Info() << "Benchmarking " << name;

    CsvFile measurements;

    for (size_t i = 0; i < repeats; i++)
    {
        auto res = cf.run();
        measurements.add(res.profiling);
    }

    saveFile(tgtFile, measurements);
}
    

// vim: set ts=4 sw=4 sts=4 expandtab:
