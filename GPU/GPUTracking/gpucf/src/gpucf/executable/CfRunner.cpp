#include "CfRunner.h"

#include <gpucf/ClusterChecker.h>
#include <gpucf/GPUClusterFinder.h>
#include <gpucf/io/ClusterWriter.h>
#include <gpucf/io/DigitReader.h>
#include <gpucf/io/DigitWriter.h>


using namespace gpucf;


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

    DigitReader dreader(digitFile->Get());

    GPUClusterFinder cf;
    auto cfRes = cf.run(env, dreader.get());

    ClusterWriter cw(clusterResultFile->Get());
    cw.write(cfRes.clusters);

    if (*peakFile)
    {
        DigitWriter dw(peakFile->Get());
        dw.write(cfRes.peaks);
    }

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
