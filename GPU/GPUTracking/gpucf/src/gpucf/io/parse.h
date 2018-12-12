#pragma once

#include <regex>
#include <string>


#define MATCH_INT(str, name) MATCH_IMPL(str, int, name, std::stoi)
#define MATCH_FLOAT(str, name) MATCH_IMPL(str, float, name, std::stof)

#define MATCH_IMPL(str, type, name, strToType) \
    type name; \
    do { \
        std::regex re(#name "\\s*=\\s*([^,\\s]*),?"); \
        std::smatch sm; \
        bool ok = std::regex_search(str, sm, re); \
        if (!ok) { \
            return false; \
        } \
        name = strToType(sm[1]); \
    } while(false)


// vim: set ts=4 sw=4 sts=4 expandtab:
