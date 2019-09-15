// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <args/args.hxx>

#include <memory>
#include <string>


namespace gpucf
{

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
    using StringFlag = args::ValueFlag<std::string>;
    using IntFlag    = args::ValueFlag<int>;

    template<typename T>
    using OptValueFlag  = std::unique_ptr<args::ValueFlag<T>>;
    using OptStringFlag = OptValueFlag<std::string>;
    using OptIntFlag    = OptValueFlag<int>;
    using OptFlag = std::unique_ptr<args::Flag>;


    virtual void setupFlags(args::Group &, args::Group &) 
    {
    }

    virtual int  mainImpl() = 0;

private:
    args::ArgumentParser parser;
    args::Group requiredArgs;
    args::Group optionalArgs;
    
};

} // namespace gpucf


// workaround because the args::Flag constructors and std::make_unique don't like
// each other
#define INIT_FLAG(type, ...) std::unique_ptr<type>(new type(__VA_ARGS__))

// vim: set ts=4 sw=4 sts=4 expandtab:
