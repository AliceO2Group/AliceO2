o2_add_executable(ex5
                  SOURCES src/run.cxx
                  COMPONENT_NAME example TARGETVARNAME targetName)

o2_target_man_page(${targetName} NAME ex5 SECTION 7)
