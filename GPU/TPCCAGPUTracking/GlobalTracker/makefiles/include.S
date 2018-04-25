    .global _makefile_opencl_program_cagpubuild_AliHLTTPCCAGPUTrackerOpenCL_cl
    .global _makefile_opencl_program_cagpubuild_AliHLTTPCCAGPUTrackerOpenCL_cl_size
    .section .rodata
_makefile_opencl_program_cagpubuild_AliHLTTPCCAGPUTrackerOpenCL_cl:
    .incbin "AliHLTTPCCAGPUTrackerOpenCLCode.bin"
1:
_makefile_opencl_program_cagpubuild_AliHLTTPCCAGPUTrackerOpenCL_cl_size:
    .int 1b - _makefile_opencl_program_cagpubuild_AliHLTTPCCAGPUTrackerOpenCL_cl
