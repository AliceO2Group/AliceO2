\page refdocManPages Man Pages

You can create man pages in nroff format under:

    Subsystem/Module/docs/<man-page-name>.<section>.in

and it will create a man page for you in:

    ${CMAKE_BINARY_DIR}/stage/share/man/man<section>

if you add:

    o2_target_man_page(target NAME <man-page-name> SECTION <section>)

to your `CMakeLists.txt`. Note the man page is "attached" to a given target.
If `SECTION` is omitted it will default to 1
(executables). For more informantion about nroff format you can look at:

    http://www.linuxjournal.com/article/1158
