#pragma once

#include <gpucf/common/Measurements.h>

#include <filesystem/path.h>

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

    void write(const filesystem::path &);
         
private:
    using Column = std::vector<std::string>;

    std::string seperator;

    std::unordered_map<std::string, Column> cols;
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
