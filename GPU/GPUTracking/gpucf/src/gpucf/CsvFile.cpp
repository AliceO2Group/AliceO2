#include "CsvFile.h"

#include <sstream>


std::string CsvFile::str() const {

    std::stringstream ss;
    for (size_t row = 0; row < colSize; row++) {
        for (size_t col = 0; col < cols.size(); col++) {
            ss << cols[col][row] << (col < cols.size()-1) ? seperator : "";
        }
        ss << std::endl;
    }

    return ss.str();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
