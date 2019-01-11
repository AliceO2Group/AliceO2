#pragma once

#include <stdexcept>


class NoPlatformFoundError : public std::runtime_error
{
public:
    NoPlatformFoundError()
        : std::runtime_error("Could not find any OpenCL platform.")
    {
    }

};

class NoGpuFoundError : public std::runtime_error
{
public:
    NoGpuFoundError()
        : std::runtime_error("Could not find any GPU devices.")
    {
    }

};

class BuildFailedError : public std::runtime_error
{
public:
    BuildFailedError()
        : std::runtime_error("Failed to build all sources.")
    {
    }
    
};





// vim: set ts=4 sw=4 sts=4 expandtab:
