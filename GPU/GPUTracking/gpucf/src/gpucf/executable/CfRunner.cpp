#include "CfRunner.h"

#include <gpucf/DataSet.h>
#include <gpucf/GPUClusterFinder.h>


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
    auto cfRes = cf.run(env, digits);

    DataSet clusterSet;

    clusterSet.serialize(cfRes.clusters);
    clusterSet.write(args::get(*clusterResultFile));

    if (*peakFile)
    {
        DataSet peaks;
        peaks.serialize(cfRes.peaks);
    }

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
