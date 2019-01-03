#include "ClEnv.h"

#include <gpucf/log.h>

#include <fstream>
#include <stdexcept>


namespace fs = filesystem;


ClEnv::ClEnv(const fs::path &srcDir, size_t gid) 
        : gpuId(gid) 
        , sourceDir(srcDir) {
    if (!sourceDir.exists() || !sourceDir.is_directory()) {
        throw std::runtime_error("Directory " + sourceDir.str()
                + " does not exist or is a file.");
    }

    cl_int err;

    err = cl::Platform::get(&platforms); 
    ASSERT(err == CL_SUCCESS);
    

    ASSERT(!platforms.empty());
    err = platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    ASSERT(err == CL_SUCCESS);

    if (devices.empty()) {
        throw std::runtime_error("Could not find any gpu devices.");
    }
    
    context = cl::Context(devices, nullptr, nullptr, nullptr, &err);
    ASSERT(err == CL_SUCCESS);
}

cl::Program ClEnv::buildFromSrc(const fs::path &srcFile) {
    cl::Program::Sources src = loadSrc(srcFile);

    cl_int err;
    cl::Program prg(context, src, &err);
    ASSERT(err == CL_SUCCESS);

    try {
        prg.build(devices);
    } catch (const cl::BuildError &) {
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }
        throw std::runtime_error("build failed.");
    }

    return prg;
}

cl::Program::Sources ClEnv::loadSrc(const fs::path &srcFile) {
    fs::path file = sourceDir / srcFile; 

    if (!file.exists()) {
        throw std::runtime_error("Could not find file " + file.str() + "."); 
    }

    std::ifstream src(file.str());
    std::string code( (std::istreambuf_iterator<char>(src)),
                      std::istreambuf_iterator<char>());

    cl::Program::Sources source({code});

    return source;
}
 
// vim: set ts=4 sw=4 sts=4 expandtab:
