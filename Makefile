#
#  Copyright (C) 1998 Jason Hutchens
#
#  This program is free software; you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the Free
#  Software Foundation; either version 2 of the license or (at your option)
#  any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the Gnu Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  675 Mass Ave, Cambridge, MA 02139, USA.
#

# DEBUG=-DDEBUG
TCLVERSION=8.5
TCLINCLUDE=-I/usr/include/tcl$(TCLVERSION)
CFLAGS=-g -Wall


all:	megahal tcllib pythonmodule perlmodule rubymodule

megahal: main.o megahal.o megahal.h backup
	gcc $(CFLAGS) -o megahal megahal.o main.o -lm $(DEBUG)
	@echo "MegaHAL is up to date"

megahal.o: megahal.c megahal.h
	gcc -fPIC $(CFLAGS) -c megahal.c

main.o: main.c megahal.h
	gcc $(CFLAGS) -c main.c

tcl-interface.o: tcl-interface.c
	gcc -fPIC $(CFLAGS) $(TCLINCLUDE) -c tcl-interface.c

tcllib: megahal.o tcl-interface.o
	gcc -fPIC -shared -Wl,-soname,libmh_tcl.so -o libmh_tcl.so megahal.o tcl-interface.o

pythonmodule: python-interface.c megahal.c
	python setup.py build -g

pythonmodule-install:
	python setup.py install --root=$(DESTDIR)

perlmodule: megahal.c megahal.h
	cd Megahal && perl Makefile.PL INSTALLDIRS=vendor && make

perlmodule-install:
	cd Megahal && make install DESTDIR=$(DESTDIR)

rubymodule:
	cd ruby && ruby extconf.rb && make

rubymodule-install:
	cd ruby && make install DESTDIR=$(DESTDIR)

version:
	./cvsversion.tcl

dist: clean version
	(cd .. ; tar -czvf megahal-`cat megahal/VERSION`.tar.gz megahal/ ; )

clean:
	rm -f megahal
	rm -rf build
	if [ -f Megahal/Makefile ]; then \
	  cd Megahal ; make clean; \
	fi
	rm -f megahal.brn megahal.log megahal.txt
	if [ -e megahal.dic.backup ];then\
		cp megahal.dic.backup megahal.dic;\
		rm megahal.dic.backup;\
	fi
	rm -f *.o *.so *~
	cd ruby && make clean

backup:
	if [ ! -e megahal.dic.backup ];then\
		cp megahal.dic megahal.dic.backup;\
	fi



#
#	$Log: Makefile,v $
#	Revision 1.8  2004/01/13 10:59:20  lfousse
#	* Applied code cleaning already shipped with the debian package.
#	* Removed pure debian stuff.
#	* Added lacking setup.py file for python module.
#	
#	Revision 1.7  2004/01/03 18:16:23  lfousse
#	Cleaned the perl directory.
#	Cleaned the Makefile.
#	Changed debian stuff (but this should not be a native package any longer).
#	
#	Revision 1.6  2003/08/26 12:49:16  lfousse
#	* Added the perl interface
#	* cleaned up the python interface a bit (but this
#	  still need some work by a python "expert")
#	* Added a learn_no_reply function.
#	
#	Revision 1.5  2002/10/16 04:32:53  davidw
#	* megahal.c (change_personality): [ 541667 ] Added patch from Andrew
#	  Burke to rework logic in change_personality.
#	
#	* megahal.c: Trailing white space cleanup.
#	
#	* python-interface.c: [ 546397 ] Change python include path to a more
#	  recent version.  This should really be fixed by using some of
#	  Python's build automation tools.
#	
#	Revision 1.4  2001/07/06 10:16:39  davidw
#	Added -fPIC for HPPA folks.
#	
#	Revision 1.3  2000/11/08 11:07:11  davidw
#	Moved README to docs directory.
#	
#	Changes to debian files.
#	
#	Revision 1.2  2000/09/07 11:43:43  davidw
#	Started hacking:
#	
#	Reduced makefile targets, eliminating non-Linux OS's.  There should be
#	a cleaner way to do this.
#	
#	Added Tcl and Python C level interfaces.
#	
#	Revision 1.2  1998/04/21 10:10:56  hutch
#	Updated.
#
#	Revision 1.1  1998/04/06 08:03:34  hutch
#	Initial revision
#
