#pragma once

#include <gpucf/common/ClEnv.h>
#include <gpucf/executable/Executable.h>


namespace gpucf
{

class Testbench : public Executable
{

public:
    Testbench();

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
