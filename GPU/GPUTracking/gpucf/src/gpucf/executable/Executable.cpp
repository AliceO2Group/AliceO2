#include "Executable.h"

#include <gpucf/cl.h>
#include <gpucf/log.h>

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
        log::Error() << "Caught cl::Error: " << err.what() << "(" << err.err() << ")";
        throw err;
    }
}

void Executable::showHelpAndExit()
{
    std::cerr << parser;    
    std::exit(1);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
