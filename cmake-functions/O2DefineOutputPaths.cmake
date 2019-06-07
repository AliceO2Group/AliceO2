include_guard()

function(o2_define_output_paths)

    # Set CMAKE_INSTALL_LIBDIR explicitly to lib (to avoid lib64 on CC7)
    set(CMAKE_INSTALL_LIBDIR lib PARENT_SCOPE)

    include(GNUInstallDirs)

    if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/stage/${CMAKE_INSTALL_BINDIR}
            PARENT_SCOPE)
    endif()
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/stage/${CMAKE_INSTALL_LIBDIR}
            PARENT_SCOPE)
    endif()
    if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/stage/${CMAKE_INSTALL_LIBDIR}
            PARENT_SCOPE)
    endif()

endfunction()
