#include "Executable.h"

#include <gpucf/log.h>

#include <iostream>


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

    return mainImpl();
}

void Executable::showHelpAndExit()
{
    std::cerr << parser;    
    std::exit(1);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
