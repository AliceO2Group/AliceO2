#pragma once

#include <gpucf/common/Digit.h>

#include <regex>
#include <string>
#include <vector>


class DigitParser 
{
    
public:
    
    bool operator()(const std::string &line, std::vector<Digit> *);

private:
    static std::regex prefix;

};

// vim: set ts=4 sw=4 sts=4 expandtab:
