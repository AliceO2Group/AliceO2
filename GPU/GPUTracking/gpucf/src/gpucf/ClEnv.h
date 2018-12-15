#pragma once

#include <gpucf/cl.h>

#include <filesystem/path.h>

#include <memory>
#include <string>
#include <vector>


class ClEnv {

public:
    ClEnv(const filesystem::path &srcDir);

    cl::Context getContext() const { return context; }
    cl::Device  getDevice()  const { return devices[0]; }

    cl::Program::Sources loadSrc(const filesystem::path &srcFile);
    
private:
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    cl::Context context;

    filesystem::path sourceDir;
};

// vim: set ts=4 sw=4 sts=4 expandtab:
