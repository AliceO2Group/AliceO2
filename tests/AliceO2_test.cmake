 ################################################################################
 #    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    #
 #                                                                              #
 #              This software is distributed under the terms of the             # 
 #         GNU Lesser General Public Licence version 3 (LGPL) version 3,        #  
 #                  copied verbatim in the file "LICENSE"                       #
 ################################################################################
Set(CTEST_SOURCE_DIRECTORY $ENV{SOURCEDIR})
Set(CTEST_BINARY_DIRECTORY $ENV{BUILDDIR})
Set(CTEST_SITE $ENV{SITE})
Set(CTEST_BUILD_NAME $ENV{LABEL})
Set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
Set(CTEST_PROJECT_NAME "AliceO2")
Set(EXTRA_FLAGS $ENV{EXTRA_FLAGS})

Find_Program(CTEST_GIT_COMMAND NAMES git)
Set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

Set(BUILD_COMMAND "make")
Set(CTEST_BUILD_COMMAND "${BUILD_COMMAND} -j$ENV{number_of_processors}")

If($ENV{ctest_model} MATCHES Nightly OR $ENV{ctest_model} MATCHES Profile)

   Find_Program(GCOV_COMMAND gcov)
   If(GCOV_COMMAND)
     Message("Found GCOV: ${GCOV_COMMAND}")
     Set(CTEST_COVERAGE_COMMAND ${GCOV_COMMAND})
   EndIf(GCOV_COMMAND)
 
   String(TOUPPER $ENV{ctest_model} _Model)
   Set(ENV{ctest_model} Nightly)

  Set(CTEST_CONFIGURE_COMMAND " \"${CMAKE_EXECUTABLE_NAME}\" \"-DCMAKE_BUILD_TYPE=${_Model}\" \"-G${CTEST_CMAKE_GENERATOR}\" \"${EXTRA_FLAGS}\" \"${CTEST_SOURCE_DIRECTORY}\" ")

  CTEST_EMPTY_BINARY_DIRECTORY(${CTEST_BINARY_DIRECTORY})

EndIf()

Configure_File(${CTEST_SOURCE_DIRECTORY}/CTestCustom.cmake
               ${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake
              )
Ctest_Read_Custom_Files("${CTEST_BINARY_DIRECTORY}")

Ctest_Start($ENV{ctest_model})
If(NOT $ENV{ctest_model} MATCHES Experimental)
  Ctest_Update(SOURCE "${CTEST_SOURCE_DIRECTORY}")
EndIf()
Ctest_Configure(BUILD "${CTEST_BINARY_DIRECTORY}")
Ctest_Build(BUILD "${CTEST_BINARY_DIRECTORY}")
Ctest_Test(BUILD "${CTEST_BINARY_DIRECTORY}" PARALLEL_LEVEL $ENV{number_of_processors})
If(${_Model} MATCHES PROFILE)
  Ctest_Coverage(BUILD "${CTEST_BINARY_DIRECTORY}")
EndIf()
Ctest_Submit()
 
