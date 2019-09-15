// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Executable.h"

#include <gpucf/common/log.h>

#include <CL/cl2.hpp>

#include <iostream>


using namespace gpucf;


int Executable::main(int argc, const char *argv[]) 
{
    args::HelpFlag help(optionalArgs, "help", "Display help menu", {'h', "help"});
    setupFlags(requiredArgs, optionalArgs);

    try 
    {
        parser.ParseCLI(argc, argv);
    } 
    catch (const args::Help &) 
    {
        showHelpAndExit();
    }
    catch (const args::ValidationError &)
    {
        log::Error() << "A required argument is missing!";
        showHelpAndExit();
    }

    try
    {
        return mainImpl();
    }
    catch(const cl::Error &err)
    {
        log::Error() << "Caught cl::Error: " << err.what() 
                     << "(" << log::clErrToStr(err.err()) << ")";
        throw err;
    }
}

void Executable::showHelpAndExit()
{
    std::cerr << parser;    
    std::exit(1);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
