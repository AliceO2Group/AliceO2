#include "Testbench.h"

#include <gpucf/debug/PeakCountTest.h>


using namespace gpucf;


Testbench::Testbench()
    : Executable("Run tests on GPU cluster finder.")
{
}

void Testbench::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags  = std::make_unique<ClEnv::Flags>(required, optional); 
}

int Testbench::mainImpl()
{
    ClusterFinderConfig config;
    ClEnv env(*envFlags, config);
    PeakCountTest pctest(config, env);

    return pctest.run(
            {{7, 0, 0, 0, 7},
             {0, 6, 0, 0, 0},
             {0, 0, 5, 0, 0},
             {0, 0, 0, 7, 0},
             {7, 0, 0, 0, 9}
            },
            {{0b11, 0, 0, 0, 0b11},
             {0},
             {0, 0, 0b10},
             {0, 0, 0, 0b10},
             {0b11, 0, 0, 0, 0b11}
            },
            {{1, 0, 0, 0, 1},
             {0},
             {0, 0, -1},
             {0, 0, 0, 1},
             {1, 0, 0, 0, 1}
            }
    );
}

// vim: set ts=4 sw=4 sts=4 expandtab:

