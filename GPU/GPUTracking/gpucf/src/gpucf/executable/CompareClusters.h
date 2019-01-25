#pragma once

#include <gpucf/executable/Executable.h>


namespace gpucf
{

class CompareClusters : public Executable
{

public:
    CompareClusters();

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    OptStringFlag truthFile;
    OptStringFlag clusterFile;
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
