#pragma once

#include "Executable.h"

#include <gpucf/ClEnv.h>


namespace gpucf
{

class SubGroupInfo : public Executable
{

public:
    SubGroupInfo();

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;
    OptIntFlag workGroupSizeFlag;
    
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
