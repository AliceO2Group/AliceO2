#pragma once

#include <filesystem/path.h>

#include <stdexcept>
#include <string>


template<bool isDirectory>
class FileError : std::exception
{

public:
    FileError(const filesystem::path &f)
        : file(f)
    {
        std::stringstream ss;
        ss << "Could not find " << ((isDirectory) ? "directory" : "file")
           << file.str() << ".";

        msg = ss.str();
    }

    const char *what() const noexcept override
    {
        return msg.c_str();
    }

private:
    std::string msg;
    filesystem::path file;

};

using FileNotFoundError      = FileError<false>;
using DirectoryNotFoundError = FileError<true>;


// vim: set ts=4 sw=4 sts=4 expandtab:
