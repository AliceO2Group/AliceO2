#include <gpucf/io/DigitReader.h>
#include <gpucf/io/BinaryWriter.h>

#include <iostream>


int main(int argc, char *argv[]) 
{

    ASSERT(argc == 2);

    std::string file = argv[1];

    log::Debug() << "Start.";


    log::Info() << "Reding text file " << file << ". This could take a while.";
    DigitReader reader(file);

    log::Info() << "Read " << reader.get().size() << " digits";

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
