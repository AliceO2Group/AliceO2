#include "Benchmark.h"

#include <gpucf/CsvFile.h>
#include <gpucf/common/log.h>
#include <gpucf/debug/StreamCompactionTest.h>
#include <gpucf/errors/FileErrors.h>
#include <gpucf/gpu/GPUClusterFinder.h>

#include <fstream>


using namespace gpucf;
namespace fs = filesystem;


Benchmark::Benchmark()
    : Executable("Run all benchmarks.")
{
}

void Benchmark::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags  = std::make_unique<ClEnv::Flags>(required, optional); 
    digitFile = OptStringFlag(
            new StringFlag(required, "FILE", "File of digits.", {'d', "digits"}));

    benchmarkDir = OptStringFlag(
            new StringFlag(required, "DIR", "Write results here.", {'o', "out"}));

    iterations = OptIntFlag(
            new IntFlag(optional, 
                        "N", 
                        "How often each algorithm in run (default=10)", 
                        {'o', "out"},
                        10));
}

int Benchmark::mainImpl()
{
    registerAlgorithms();

    ClEnv env(*envFlags); 

    /* StreamCompactionTest streamCompactionTest(env); */
    /* streamCompactionTest.run(); */

    DataSet digitSet;
    digitSet.read(args::get(*digitFile));

    setupAlgorithms(env, digitSet);

    baseDir = benchmarkDir->Get();

    if (!baseDir.is_directory())
    {
        throw DirectoryNotFoundError(baseDir);
    }

    run(iterations->Get());

    return 0;
}

void Benchmark::registerAlgorithms()
{
    algorithms.emplace_back(new GPUClusterFinder);
}

void Benchmark::setupAlgorithms(ClEnv &env, const DataSet &data)
{
    for (auto &algorithm : algorithms)
    {
        algorithm->setup(env, data);
    }
}

void Benchmark::run(size_t N)
{
    for (auto &algorithm : algorithms)
    {
        std::string algName = algorithm->getName();
        log::Info() << "Benchmarking " << algName << "...";

        CsvFile measurements;
        for (size_t i = 0; i < N; i++)
        {
            auto res = algorithm->run();
            measurements.add(res.profiling);
        }

        fs::path tgtFile = makeBenchmarkFilename(algName);
        log::Info() << "Writing results to " << tgtFile;

        std::ofstream out(tgtFile.str());
        out << measurements.str();
    }
}


fs::path Benchmark::makeBenchmarkFilename(const std::string &algName)
{
    fs::path fname = baseDir / fs::path(algName + ".csv");    
    return fname;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
