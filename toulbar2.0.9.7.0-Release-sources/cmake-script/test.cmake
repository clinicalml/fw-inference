#INCLUDE(CTest)


    file ( GLOB_RECURSE
                        validation_file
                       validation/*.wcsp
                                    )

###################"
#################
# test unitaire
###############
SET(FOPT "test-opt.cmake") #cmake name where are declared local value for timeout,regexp and command line option

	MESSAGE(STATUS "##############TEST liste building #############")
FOREACH (UTEST ${validation_file})
	#reset ub end enum from the previous iteration
        UNSET(UB) 
	UNSET(ENUM)


	STRING(REPLACE ".wcsp" ".ub" UBF ${UTEST})
	STRING(REPLACE ".wcsp" ".enum" ENUM_file ${UTEST})
	
	GET_FILENAME_COMPONENT(TPATH ${UTEST} PATH  )

	IF (EXISTS ${UBF})
	FILE(READ ${UBF} UB)
	STRING(REPLACE "\n" ""  TUB ${UB})
	SET (UB ${TUB})
	MATH(EXPR UBP "1+${TUB}")
        SET (OpTub "-ub" ) 
        SET (UBP "${OpTub}=${UBP}" )	
	MESSAGE(STATUS "UB found ==> ${UBP} " )

	ELSE()
	 UNSET(UBP)
	ENDIF()

	 IF (EXISTS ${ENUM_file})
        FILE(READ ${ENUM_file} TENUM)
	STRING(REPLACE "\n" ""  ENUM ${TENUM})
	MESSAGE(STATUS "ENUM file: ${TPATH}/${ENUM_file} founded ENUM variable loaded ")
	ELSE()
	set(ENUM)
        ENDIF()

	
	IF (EXISTS ${TPATH}/${FOPT})
	include (${TPATH}/${FOPT})
	MESSAGE(STATUS "file: ${TPATH}/${FOPT} founded ")
	ELSE()

	# init default value :

	set (command_line_option ${Default_test_option} )
	set (test_timeout ${Default_test_timeout})
	set (test_regexp  ${Default_test_regexp})
	
	MESSAGE(STATUS "file: ${TPATH}/${FOPT} not founded  ==>
	default option used: command line : ${command_line_option} timeout=${test_timeout};regexp=${test_regexp} ")
	ENDIF()	


	MESSAGE(STATUS "file: ${UTEST} used opt = ${command_line_option}")


	STRING(REPLACE "${PROJECT_SOURCE_DIR}/validation/" "" TMP ${UTEST})
	STRING(REPLACE ".wcsp" ""  TNAME ${TMP})

	if($verbose) 
		MESSAGE(STATUS "UBF: ${UBF}")
		MESSAGE(STATUS "UB: ${UB}")
		MESSAGE(STATUS "TNAME: ${TNAME}")
	endif($verbose)

	IF (EXISTS ${UBF}) # if ub file exist
		add_test(Phase1_Toulbar_${TNAME} ${EXECUTABLE_OUTPUT_PATH}/toulbar2${EXE}  ${UTEST} ${UBP} ${command_line_option})
		set_tests_properties (Phase1_Toulbar_${TNAME} PROPERTIES PASS_REGULAR_EXPRESSION "${test_regexp}" TIMEOUT "${test_timeout}")
	ELSE()
		add_test(Phase1_Toulbar_${TNAME} ${EXECUTABLE_OUTPUT_PATH}/toulbar2${EXE}  ${UTEST} ${command_line_option})
		set_tests_properties (Phase1_Toulbar_${TNAME} PROPERTIES PASS_REGULAR_EXPRESSION "${test_regexp}" TIMEOUT "${test_timeout}")
	ENDIF()


ENDFOREACH(UTEST)

#	MESSAGE(STATUS "\n")

ENABLE_TESTING()
###################"
