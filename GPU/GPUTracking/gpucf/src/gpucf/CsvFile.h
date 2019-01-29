#pragma once

#include <gpucf/common/Measurements.h>

#include <string>
#include <unordered_map>
#include <vector>


namespace gpucf
{

class CsvFile 
{

public:
    CsvFile(const std::string &sep=",");

    void add(const Measurements &);

    std::string str() const;

private:
    struct Column
    {
        std::string head;
        std::vector<std::string> entries;

        Column(const std::string &);
    };

    std::string seperator;

    std::unordered_map<std::string, size_t> columnLookup; 
    std::vector<Column> columns;

    std::string suffix(size_t) const;

};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
