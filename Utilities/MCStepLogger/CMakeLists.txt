# @author Sandro Wenzel
# @brief  cmake setup for module Utilities/MCStepLogger

set(MODULE_NAME "MCStepLogger")
set(MODULE_BUCKET_NAME mcsteplogger_bucket)
O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/MCStepInterceptor.cxx
    src/MCStepLoggerImpl.cxx
   )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()
