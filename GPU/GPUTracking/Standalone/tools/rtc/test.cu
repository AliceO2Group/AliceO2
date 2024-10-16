#include <cuda.h>
#include <nvrtc.h>
#include <iostream>

#define NVRTC_SAFE_CALL(x)                              \
  do {                                                  \
    nvrtcResult result = x;                             \
    if (result != NVRTC_SUCCESS) {                      \
      std::cerr << "\nerror: " #x " failed with error " \
                << nvrtcGetErrorString(result) << '\n'; \
      exit(1);                                          \
    }                                                   \
  } while (0)

#define CUDA_SAFE_CALL(x)                               \
  do {                                                  \
    CUresult result = x;                                \
    if (result != CUDA_SUCCESS) {                       \
      const char* msg;                                  \
      cuGetErrorName(result, &msg);                     \
      std::cerr << "\nerror: " #x " failed with error " \
                << msg << '\n';                         \
      exit(1);                                          \
    }                                                   \
  } while (0)

int32_t main(int argc, char** argv)
{
  //Read Sourcecode from file
  uint32_t filesize;
  FILE* pFile;
  //Open file
  if ((pFile = fopen("source.cu", "rb")) == NULL)
    exit(1);
  //Optain File Size
  fseek(pFile, 0, SEEK_END);
  filesize = ftell(pFile);
  rewind(pFile);
  //Read file
  char* sourceCode = new char[filesize + 1];
  if (fread(sourceCode, 1, filesize, pFile) != filesize)
    exit(1);
  //Make sourceCode 0-terminated
  sourceCode[filesize] = 0;
  fclose(pFile);

  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,      // prog
                                     sourceCode, // buffer
                                     "saxpy.cu", // name
                                     0,          // numHeaders
                                     NULL,       // headers
                                     NULL));     // includeNames
  delete[] sourceCode;
  //const char *opts[] = {"-default-device -std=c++17  --extended-lambda -Xptxas -O4 -Xcompiler -O4 -use_fast_math --ftz=true"};
  const char* opts[] = {"-default-device", "--std=c++17", "-use_fast_math", "-ftz=true"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,                           // prog
                                                  sizeof(opts) / sizeof(opts[0]), // numOptions
                                                  opts);                          // options
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char* log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS) {
    exit(1);
  }
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char* ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  CUmodule module;
  CUfunction kernel;
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "foo"));
  void* args[] = {};
  CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
                   1, 1, 1,   // grid dim
                   32, 1, 1,  // block dim
                   0, NULL,   // shared mem and stream
                   args, 0)); // arguments
  return 0;
}
