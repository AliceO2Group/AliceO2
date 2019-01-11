#pragma once

#include <gpucf/Executable.h>
#include <gpucf/ClEnv.h>

#include <memory>


class ClCanary : public Executable 
{
    
public:
    ClCanary() 
        : Executable("Tests if OpenCL is working as expected.") 
    {
    }

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;

};

// vim: set ts=4 sw=4 sts=4 expandtab:
