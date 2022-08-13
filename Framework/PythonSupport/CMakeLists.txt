
add_library(o2 SHARED src/o2module.cc)

target_link_libraries(o2 Python::Module)
target_include_directories(o2 PUBLIC 
                                ../../Common/MathUtils/include 
                                ../../GPU/Common
                                ../../Common/Constants/include
                                ../Core/include ../Foundation/include)
add_compile_definitions(DPL_CUSTOM_DATA_MODEL)


set_target_properties(
    o2 
    PROPERTIES
        PREFIX ""
        SUFFIX .so
        OUTPUT_NAME "o2"
        LINKER_LANGUAGE C
    )

