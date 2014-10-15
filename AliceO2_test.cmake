 ################################################################################
 #    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    #
 #                                                                              #
 #              This software is distributed under the terms of the             # 
 #         GNU Lesser General Public Licence version 3 (LGPL) version 3,        #  
 #                  copied verbatim in the file "LICENSE"                       #
 ################################################################################
SET (CTEST_SOURCE_DIRECTORY $ENV{SOURCEDIR})
SET (CTEST_BINARY_DIRECTORY $ENV{BUILDDIR})
SET (CTEST_SITE $ENV{SITE})
SET (CTEST_BUILD_NAME $ENV{LABEL})
SET (CTEST_CMAKE_GENERATOR "Unix Makefiles")
SET (CTEST_PROJECT_NAME "FAIRROOT")

Find_program(CTEST_GIT_COMMAND NAMES git)
Set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

#If($ENV{ctest_model} MATCHES Continuous)
#  Set(CTEST_SVN_UPDATE_OPTIONS "$ENV{REVISION}")
#EndIf($ENV{ctest_model} MATCHES Continuous)

SET (BUILD_COMMAND "make")
SET (CTEST_BUILD_COMMAND "${BUILD_COMMAND} -j$ENV{number_of_processors}")

If($ENV{ctest_model} MATCHES Nightly)

Set (CTEST_CONFIGURE_COMMAND " \"${CMAKE_EXECUTABLE_NAME}\" \"-DCMAKE_BUILD_TYPE=NIGHTLY\" \"-G${CTEST_CMAKE_GENERATOR}\" \"${CTEST_SOURCE_DIRECTORY}\" ")

  # get the information about conflicting or localy modified files
  # from svn, extract the relavant information about the file name
  # and put the result in the output variable
#  execute_process(COMMAND svn stat -u  
#                  COMMAND grep ^[CM]
#                  COMMAND cut -c21- 
#                  OUTPUT_VARIABLE FILELIST
#                  )

  # create out of the output a cmake list. This step is done to convert the
  # stream into seperated filenames.
  # The trick is to exchange an "\n" by an ";" which is the separartor in
  # a list created by cmake 
#  STRING(REGEX REPLACE "\n" ";" _result "${FILELIST}")

#  FOREACH(_file ${_result})
#    STRING(STRIP "${_file}" _file1)
#    SET (CTEST_NOTES_FILES ${CTEST_NOTES_FILES} "${CTEST_SOURCE_DIRECTORY}/${_file1}")
#  ENDFOREACH(_file ${_result})

  CTEST_EMPTY_BINARY_DIRECTORY(${CTEST_BINARY_DIRECTORY})

endif($ENV{ctest_model} MATCHES Nightly)

configure_file(${CTEST_SOURCE_DIRECTORY}/CTestCustom.cmake
               ${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake
              )
ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")

CTEST_START ($ENV{ctest_model})
CTEST_UPDATE (SOURCE "${CTEST_SOURCE_DIRECTORY}")
CTEST_CONFIGURE (BUILD "${CTEST_BINARY_DIRECTORY}")
CTEST_BUILD (BUILD "${CTEST_BINARY_DIRECTORY}")
CTEST_TEST (BUILD "${CTEST_BINARY_DIRECTORY}" PARALLEL_LEVEL $ENV{number_of_processors})
CTEST_SUBMIT ()
 
