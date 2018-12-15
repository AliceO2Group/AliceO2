#include "ClEnv.h"

#include <gpucf/log.h>

#include <fstream>
#include <stdexcept>


namespace fs = filesystem;


ClEnv::ClEnv(const fs::path &srcDir) 
        : sourceDir(srcDir) {
    if (!sourceDir.exists() || !sourceDir.is_directory()) {
        throw std::runtime_error("Directory " + sourceDir.str()
                + " does not exist or is a file.");
    }

    cl::Platform::get(&platforms); 
    
    ASSERT(!platforms.empty());
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        throw std::runtime_error("Could not find any gpu devices.");
    }
    
    context = cl::Context(devices);
}

cl::Program::Sources ClEnv::loadSrc(const fs::path &srcFile) {
    fs::path file = sourceDir / srcFile; 

    if (!file.exists()) {
        throw std::runtime_error("Could not find file " + file.str() + "."); 
    }

    std::ifstream src(file.str());
    std::string code( (std::istreambuf_iterator<char>(src)),
                      std::istreambuf_iterator<char>());

    cl::Program::Sources source(1, std::make_pair(code.c_str(),
        code.length() + 1));

    return source;
}
 
// vim: set ts=4 sw=4 sts=4 expandtab:
