#pragma once

#include <chrono>


namespace gpucf
{

class Timer
{

public:
    void begin()
    {
        start = now();
    }

    void stop()
    {
        end = now();
    }

    float elapsedTime() const
    {
        std::chrono::duration<float> elapsed = end - start;
        return elapsed.count();
    }

private:
    using tpoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

    tpoint start;
    tpoint end;

    tpoint now()
    {
        return std::chrono::high_resolution_clock::now();
    }

    
};
    
};

// vim: set ts=4 sw=4 sts=4 expandtab:
