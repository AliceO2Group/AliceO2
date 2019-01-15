#pragma once

#include <string>
#include <vector>


namespace gpucf
{

class CsvFile 
{

public:
    CsvFile(char sep=',') 
        : seperator(sep) 
    {
    }

    template<typename T>
    void addColumn(const std::string &name, const std::vector<T> &vals) 
    {

        if (vals.empty()) 
        {
            throw std::runtime_error("Csv columns can not be empty.");
        }

        cols.emplace_back(); 

        auto &newCol = cols.back();

        if (colSize == 0) 
        {
            colSize = vals.size() + 1;
        }

        if (colSize != vals.size() + 1) 
        {
            throw std::runtime_error("New column has the wrong size.");    
        }

        newCol.reserve(colSize);

        newCol.push_back(name);
        for (const T &val : vals) 
        {
            newCol.push_back(std::to_string(val));    
        }
    }

    std::string str() const;
         
private:
    using Column = std::vector<std::string> values;

    std::string seperator;
    size_t colSize = 0;

    std::vector<Column> cols;

};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
