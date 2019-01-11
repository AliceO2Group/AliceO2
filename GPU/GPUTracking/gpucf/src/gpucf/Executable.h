#pragma once

#include <args/args.hxx>

#include <string>


class Executable 
{

public:
    Executable(const std::string &desc) 
        : parser(desc) 
        , requiredArgs(parser, "Required Arguments", args::Group::Validators::All)
        , optionalArgs(parser, "Optional Arguments")
    {
    }

    virtual ~Executable() 
    {
    }

    int main(int argc, const char *argv[]);

    void showHelpAndExit();

protected:
    virtual void setupFlags(args::Group &, args::Group &) 
    {
    }

    virtual int  mainImpl() = 0;

private:
    args::ArgumentParser parser;
    args::Group requiredArgs;
    args::Group optionalArgs;
    
};

// vim: set ts=4 sw=4 sts=4 expandtab:
