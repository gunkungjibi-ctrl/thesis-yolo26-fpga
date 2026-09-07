# build libpreproc_xrt.so บนบอร์ด KV260 (PetaLinux 2022.2 มี XRT header ที่ /usr/include/xrt)
#   make            -> libpreproc_xrt.so
#   make check      -> compile-only check (ใช้บน host ที่ไม่มี XRT ไม่ได้ — ต้องมี header)
XRT_INC ?= /usr/include/xrt
XRT_LIB ?= /usr/lib
CXX     ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fPIC -Wall -I$(XRT_INC) -I../hls

libpreproc_xrt.so: preproc_xrt.cpp ../hls/preproc_accel.h
	$(CXX) $(CXXFLAGS) -shared -o $@ preproc_xrt.cpp -L$(XRT_LIB) -lxrt_coreutil -lpthread

clean:
	rm -f libpreproc_xrt.so
.PHONY: clean
