# list of arguments use in command line for the current directory
set (command_line_option -B=0 -v -e: )  
# test timeout ( used for all wcsp founded in the directory
set (test_timeout 100)
#regexp to define successfull end.
set (test_regexp  "Optimum: ${UB}")

#regex error can also be define: ...add set_test_propertie in test.cmake ...to be done
