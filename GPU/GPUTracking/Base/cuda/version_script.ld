{
  global:
    extern "C++" {
      "GPUReconstruction_Create_CUDA";
      "o2::gpu::cuda::*"
    };
  local:
    extern "C++" {
      o2::*;
    };
};
