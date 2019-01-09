#pragma once

#include <args/args.hxx>

#include <string>


class Executable 
{

public:
    Executable(const std::string &desc) 
        : parser(desc) 
    {
    }

    virtual ~Executable() 
    {
    }

    int main(int argc, const char *argv[]);

protected:
    virtual void setupFlags(args::ArgumentParser &) 
    {
    }

    virtual int  mainImpl() = 0;

private:
    args::ArgumentParser parser;
    
};

// vim: set ts=4 sw=4 sts=4 expandtab:
