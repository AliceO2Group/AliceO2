#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>

#include <string>


namespace gpucf
{

std::string clErrToStr(cl_int);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
