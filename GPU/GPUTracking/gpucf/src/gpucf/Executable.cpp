#include "Executable.h"

#include <iostream>


int Executable::main(int argc, const char *argv[]) 
{
    args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
    setupFlags(parser);

    try 
    {
        parser.ParseCLI(argc, argv);
    } 
    catch (const args::Help &) 
    {
        std::cout << parser;
        std::exit(0);
    }

    return mainImpl();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
