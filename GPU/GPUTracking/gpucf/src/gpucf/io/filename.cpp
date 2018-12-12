#include "filename.h"

#include <gpucf/log.h>

#include <regex>


bool isBinary(const std::string &name) {
    return getExt(name) == "bin";
}

bool isText(const std::string &name) {
    return getExt(name) == "txt";
}

std::string getExt(const std::string &name) {
    return name.substr(name.find_last_of(".") + 1);
}

std::string getHead(const std::string &name) {
    return name.substr(0, name.find_last_of("."));
}

// vim: set ts=4 sw=4 sts=4 expandtab:
