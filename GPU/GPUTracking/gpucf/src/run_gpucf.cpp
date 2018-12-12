#include <gpucf/io/DigitReader.h>
#include <gpucf/io/filename.h>
#include <gpucf/io/BinaryWriter.h>

#include <iostream>


int main(int argc, char *argv[]) {

    ASSERT(argc == 2);

    std::string file = argv[1];

    log::Debug() << "Start.";

    std::string head = getHead(file);
    std::string binfile = head + ".bin";

    log::Debug() << "binfile = " << binfile;


    log::Info() << "Reding text file " << file << ". This could take a while.";
    DigitReader reader(file);

    log::Info() << "Read " << reader.get().size() << " digits";

    log::Info() << "Writing to binary file " << binfile;
    BinaryWriter writer(binfile);
    writer.write(reader.get());

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
