#include "CsvFile.h"

#include <gpucf/common/log.h>

#include <sstream>


using namespace gpucf;


CsvFile::Column::Column(const std::string &name)
    : head(name)
{
}

CsvFile::CsvFile(const std::string &sep)
    : seperator(sep)
{
}

void CsvFile::add(const Measurements &measurements)
{
    for (const auto &m : measurements)
    {
        auto colRef = columnLookup.find(m.first);
        if (colRef == columnLookup.end())
        {
            columnLookup[m.first] = columns.size();
            columns.emplace_back(m.first);
        }
        columns[columnLookup[m.first]].entries.push_back(
                std::to_string(m.second));
    }
}

std::string CsvFile::str() const 
{
    std::stringstream header;
    for (size_t c = 0; c < columns.size(); c++) 
    {
        header << columns[c].head << suffix(c);
    }

    ASSERT(!columns.empty());
    // should take the maximum size of 
    // all columns here
    // but meh ¯\_(ツ)_/¯
    size_t numOfRows = columns[0].entries.size(); 

    std::stringstream entries;
    for (size_t r = 0; r < numOfRows; r++)
    {
        for (size_t c = 0; c < columns.size(); c++)
        {
            entries << columns[c].entries[r] << suffix(c);
        }
    }

    header << entries.str();
    return header.str();
}

std::string CsvFile::suffix(size_t column) const
{
    return (column < columns.size()-1 ? seperator : "\n");
}

// vim: set ts=4 sw=4 sts=4 expandtab:
