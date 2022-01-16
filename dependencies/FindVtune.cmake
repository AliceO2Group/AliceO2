if (DEFINED ENV{O2_ENABLE_VTUNE})
  set(ENABLE_VTUNE_PROFILER TRUE)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(Vtune REQUIRED IMPORTED_TARGET ittnotify)

endif()
