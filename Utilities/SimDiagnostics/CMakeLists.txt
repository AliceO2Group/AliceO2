# @author Benedikt Volkel
# @brief  cmake setup for module Utilities/SimDiagnostics

set(MODULE_NAME "SimDiagnostics")

O2_SETUP(NAME ${MODULE_NAME})

set(BUCKET_NAME simdiagnostics_bucket)

set(SRCS
    src/SimInterceptorImpl.cxx
    )

set(LIBRARY_NAME ${MODULE_NAME})

O2_GENERATE_LIBRARY()
