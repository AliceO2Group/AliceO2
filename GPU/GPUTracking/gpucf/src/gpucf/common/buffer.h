#pragma once

#include <nonstd/span.hpp>

#include <CL/cl2.hpp>


namespace gpucf
{

enum class Memory : cl_mem_flags
{
    ReadOnly  = CL_MEM_READ_ONLY,
    WriteOnly = CL_MEM_WRITE_ONLY,
    ReadWrite = CL_MEM_READ_WRITE,
};


template<typename T>
cl::Buffer makeBuffer(size_t N, Memory mem, cl::Context context)
{
    size_t bytes = sizeof(T) * N;
    
    return cl::Buffer(context, static_cast<cl_mem_flags>(mem), bytes);
}

template<typename T>
void gpucpy(
        nonstd::span<const T> from, 
        cl::Buffer to, 
        size_t N, 
        cl::CommandQueue queue,
        bool blocking=false)
{
    size_t bytes = sizeof(T) * N; 

    queue.enqueueWriteBuffer(
        to,
        blocking,
        0,
        bytes,
        from.data());
}

template<typename T>
void gpucpy(
        cl::Buffer from,
        nonstd::span<T> to,
        size_t N,
        cl::CommandQueue queue,
        bool blocking=false,
        size_t offset=0)
{
    size_t bytes = sizeof(T) * N; 

    queue.enqueueReadBuffer(
        from,
        blocking,
        offset * sizeof(T),
        bytes,
        to.data());
}

template<typename T>
void fill(cl::Buffer buf, T val, size_t N, cl::CommandQueue queue)
{
    queue.enqueueFillBuffer(buf, val, 0, N * sizeof(T)); 
}


} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
