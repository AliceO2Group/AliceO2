#include "CfRunner.h"

#include <gpucf/common/DataSet.h>
#include <gpucf/common/log.h>
#include <gpucf/gpu/GPUClusterFinder.h>


using namespace gpucf;


CfRunner::CfRunner()
    : Executable("Runs the GPU cluster finder.")
{
}

void CfRunner::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags  = std::make_unique<ClEnv::Flags>(required, optional); 
    digitFile = OptStringFlag(
            new StringFlag(required, "FILE", "File of digits.", {'d', "digits"}));

    clusterResultFile = OptStringFlag(
            new StringFlag(required, "FILE", "Write results clusters here.", {'o', "out"}));

    peakFile = OptStringFlag(
            new StringFlag(optional, "FILE", "Write cluster peaks here.", {'p', "peaks"}));
}

int CfRunner::mainImpl()
{
    ClEnv env(*envFlags); 

    DataSet digitSet;
    digitSet.read(args::get(*digitFile));

    std::vector<Digit> digits = digitSet.deserialize<Digit>();

    GPUClusterFinder cf;
    cf.setup(GPUClusterFinder::defaultConfig, env, digits);
    auto cfRes = cf.run();

    DataSet clusters;
    clusters.serialize(cfRes.clusters);
    clusters.write(args::get(*clusterResultFile));

    if (*peakFile)
    {
        auto peaks = cf.getPeaks();
        log::Info() << "Writing " << peaks.size()
                    << " peaks to file " << peakFile->Get();
        DataSet peakSet;
        peakSet.serialize<Digit>(peaks);
        peakSet.write(peakFile->Get());
    }

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
