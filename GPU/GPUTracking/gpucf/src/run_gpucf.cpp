#include <gpucf/io/DigitReader.h>

#include <iostream>


int main(int argc, char *argv[]) {

    ASSERT(argc == 2);

    DigitReader reader(argv[1]);

    log::Info() << reader.get().size();
    /* for (const Digit &d : reader.get()) { */
    /*     std::cout << d << std::endl; */ 
    /* } */

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
