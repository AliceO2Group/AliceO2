#include <args/args.hxx>

#include <iostream>


int main(int argc, const char *argv[]) {
    args::ArgumentParser parser("Tets for args lib"); 
    args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
    args::Flag foo(parser, "foo", "Foo Flag", {'f', "foo"});
    args::ValueFlag<int> bar(parser, "bar", "Bar Flag", {'b', "bar"});

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help &) {
        std::cout << parser;
        std::exit(0);
    }

    if (foo) {
        std::cout << "Foo flag set." << std::endl;
    }

    if (bar) {
        std::cout << "bar = " << args::get(bar) << std::endl;
    }

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:

