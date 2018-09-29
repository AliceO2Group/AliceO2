#include "log.h"

int main() {
    log::Debug() << "This is debug message.";
    log::Info() << "This is a info message";
    log::Success() << "Something good happened!";
    log::Fail() << "HALT AND CATCH FIRE";
    log::Info() << "You shouldn't see this";

    return 0;
}
