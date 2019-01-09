#pragma once

#include <fstream>
#include <vector>


class BinaryReader 
{
    
public:
    BinaryReader(const std::string &fName) 
        : file(fName, std::ios::ate | std::ios::binary)
    {
    }
        
    template<typename T>
    std::vector<T> read() 
    {
        size_t pos = file.tellg();

        if (pos % sizeof(T) == 0) 
        {
            throw std::runtime_error("Filesize doesn't match requested type.");
        }

        std::vector<T> data(pos / sizeof(T));

        file.seek(0, ios:beg);
        file.read(reinterpret_cast<char *>(data.data()), pos);

        return data;
    }

private:
    std::fstream file;

};

// vim: set ts=4 sw=4 sts=4 expandtab:
