#pragma once

#include <nonstd/span.hpp>

#include <fstream>


class BinaryWriter 
{
    
public:
    BinaryWriter(const std::string &fName) 
        : file(fName, std::ios::out | std::ios::binary)
    {
    }
        
    template<typename T>
    void write(const std::vector<T> &data) 
    {
        
        size_t bytes = sizeof(T) * data.size();
        
        file.write(reinterpret_cast<const char *>(data.data()), bytes);
    }

private:
    std::fstream file;

};

// vim: set ts=4 sw=4 sts=4 expandtab:
