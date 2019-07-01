#include "TimeCf.h"

#include <gpucf/algorithms/ClusterFinderProfiler.h>
#include <gpucf/algorithms/ClusterFinderTest.h>
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
    : Experiment(baseDir, conf)
    , name(n)
    , tgtFile(tgt)
    , repeats(N)
    , digits(d)
{
}

void TimeCf::run(ClEnv &env)
{

    log::Info() << "Benchmarking " << name << "(" << cfg << ")";

    ClusterFinderTest tester(cfg, digits.size(), env);
    tester.run(digits);

    Measurements measurements;

    for (size_t i = 0; i < repeats+1; i++)
    {
        ClEnv envCopy = env;

        ClusterFinderProfiler p(cfg, digits.size(), envCopy);
        auto res = p.run(digits);

        if (i > 0)
        {
            measurements.add(res);
            measurements.finishRun();
        }
    }

    save(tgtFile, measurements);
}
    

// vim: set ts=4 sw=4 sts=4 expandtab:
