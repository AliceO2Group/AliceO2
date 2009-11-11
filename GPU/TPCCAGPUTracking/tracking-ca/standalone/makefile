all:		caexecutable

calibrary:
			+export CONFIGFILE=config_lib.mak && make -f makefile_tmp

caexecutable:	calibrary
			+make -f makefile_tmp
			
clean:
			+make -f makefile_tmp clean