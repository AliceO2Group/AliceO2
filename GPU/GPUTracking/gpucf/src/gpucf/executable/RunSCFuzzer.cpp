#include "RunSCFuzzer.h"

#include <gpucf/debug/SCFuzzer.h>


using namespace gpucf;


RunSCFuzzer::RunSCFuzzer()
    : Executable("Fuzzy-ish test of StreamCompaction for race conditions.")
{
}

void RunSCFuzzer::setupFlags(
        args::Group &required, 
        args::Group &optional)
{
    envFlags  = std::make_unique<ClEnv::Flags>(required, optional); 

    numRuns = OptIntFlag(
            new IntFlag(required,
                        "N",
                        "How often the StreamCompaction is run.",
                        {'r', "run"}));            
}

int RunSCFuzzer::mainImpl()
{
    ClEnv env(*envFlags);

    SCFuzzer fuzzer(env);

    fuzzer.run(numRuns->Get());

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:

