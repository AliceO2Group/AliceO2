#pragma once

#include <gpucf/ClEnv.h>
#include <gpucf/executable/Executable.h>
#include <gpucf/gpu/GPUAlgorithm.h>


namespace gpucf
{
    
class Benchmark : public Executable
{

public:
    Benchmark();

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;
    OptStringFlag digitFile;
    OptStringFlag benchmarkDir;
    OptIntFlag iterations;

    filesystem::path baseDir;
    std::vector<std::shared_ptr<GPUAlgorithm>> algorithms;

    void registerAlgorithms();

    void setupAlgorithms(ClEnv &, const DataSet &);

    void run(size_t);

    filesystem::path makeBenchmarkFilename(const std::string &);

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
