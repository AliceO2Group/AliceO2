#pragma once

#include <gpucf/Executable.h>
#include <gpucf/ClEnv.h>
#include <gpucf/io/DigitReader.h>

#include <memory>


class CfRunner : public Executable
{
    
public:
    CfRunner()
        : Executable("Runs the GPU cluster finder.")
    {
    }

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;
    std::unique_ptr<DigitReader::Flags> digitFlags;
};

// vim: set ts=4 sw=4 sts=4 expandtab:

