You can create man pages in nroff format under:

    Subsystem/Module/docs/<man-page-name>.<section>.in

and it will create a man page for you in:

    ${CMAKE_BINARY_DIR}/share/man/man<section>

if you add:

    O2_GENERATE_MAN(NAME <man-page-name> SECTION <section>)

to your `CMakeLists.txt`. If `SECTION` is omitted it will default to 1
(executables). For more informantion about nroff format you can look at:

    http://www.linuxjournal.com/article/1158
