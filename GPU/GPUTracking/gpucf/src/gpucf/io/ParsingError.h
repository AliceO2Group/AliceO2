#pragma once

#include <stdexcept>
#include <sstream>


class ParsingError {

public:
    ParsingError(const std::string &f, size_t l)
        : file(f)
        , line(l)
    {}

    const char *what() const noexcept {
        std::stringstream ss; 
        ss << "Error parsing file " << file << " in line " << line;
        return ss.str().c_str();
    }

private:
    std::string file;
    size_t line;
    
};

// vim: set ts=4 sw=4 sts=4 expandtab:
