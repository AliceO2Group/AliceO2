#include "CsvFile.h"

#include <sstream>


using namespace gpucf;


CsvFile::CsvFile(const std::string &sep)
    : seperator(sep)
{
}

void CsvFile::add(const Measurements &measurements)
{
    for (const auto &m : measurements)
    {
        cols[m.first].push_back(m.second);
    }
}

std::string CsvFile::str() const 
{
    /* std::stringstream entries; */
    std::stringstream header;
    for (const auto &col : cols) 
    {
        header << col.first << " ";
    }
    header << std::endl;

    return header.str();
}

void CsvFile::write(const fs::path &tgt) const
{
    log::Warn() << __func__ << " not implemented yet!";
}

// vim: set ts=4 sw=4 sts=4 expandtab:
