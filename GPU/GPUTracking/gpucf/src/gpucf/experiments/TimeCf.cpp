#include "TimeNaiveCf.h"

#include <gpucf/common/log.h>


using namespace gpucf;
namespace fs = filesystem;


TimeCf::TimeCf(
        GPUClusterFinder::Config conf,
        nonstd::span<const Digit> d, 
        size_t N, 
        fs::path baseDir)
    : Experiment(baseDir)
    , config(conf)
    , repeats(N)
    , digits(d)
{
}

void TimeCf::run(ClEnv &env)
{
    GPUClusterFinder cf; 

    cf.setup(GPUClusterFinder::defaultConfig, env, digits);

    log::Info() << "Benchmarking naive cluster finder...";

    CsvFile measurements;

    for (size_t i = 0; i < repeats; i++)
    {
        auto res = cf.run();
        measurements.add(res.profiling);
    }

    saveFile("naiveClusterFinder.csv", measurements);
}
    

// vim: set ts=4 sw=4 sts=4 expandtab:
