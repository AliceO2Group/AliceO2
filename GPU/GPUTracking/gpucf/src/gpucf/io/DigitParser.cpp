#include "DigitParser.h"

#include "parse.h"

#include <gpucf/log.h>


std::regex DigitParser::prefix = std::regex("Digit:\\s*(.*)");


bool DigitParser::operator()(const std::string &line, 
        std::vector<Digit> *digits) {
    std::smatch sm; 
    bool isDigit = std::regex_match(line, sm, prefix);

    log::Debug() << line;

    if (!isDigit) {
        return true;
    }
    ASSERT(sm.size() == 2);

    const std::string &digitMembers = sm[1];    

    MATCH_FLOAT(digitMembers, charge);
    MATCH_INT(digitMembers, cru);
    MATCH_INT(digitMembers, row);
    MATCH_INT(digitMembers, pad);
    MATCH_INT(digitMembers, time);

    digits->emplace_back(charge, cru, row, pad, time);
    return true;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
