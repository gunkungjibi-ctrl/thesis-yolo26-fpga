# build/run C testbench ของ preproc_accel ด้วย g++ (ไม่ต้องมี Vitis HLS)
#   make vectors   -> gen test vector ด้วย golden model (ต้องมี numpy; cv2ถ้ามีจะเทียบให้ด้วย)
#   make test      -> compile tb + รันเทียบ golden ทุกเคส
#   make hls       -> vitis_hls -f run_hls.tcl (csim + csynth + export .xo)  [ต้องมี Vitis HLS 2022.2]
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++11 -Wall -Wno-unknown-pragmas -Wno-unused-label
VEC      ?= vectors
SIZES    ?= 640x360 640x640 1280x720 1920x1080 1032x582
PY       ?= python3

tb_preproc: preproc_accel.cpp tb_preproc.cpp preproc_accel.h
	$(CXX) $(CXXFLAGS) -o $@ preproc_accel.cpp tb_preproc.cpp

vectors:
	@mkdir -p $(VEC)
	@for s in $(SIZES); do $(PY) ../golden/preproc_golden.py gen $(VEC) --src $$s; done

test: tb_preproc
	@test -f $(VEC)/meta_640x360.txt || $(MAKE) vectors
	./tb_preproc $(VEC)

hls:
	vitis_hls -f run_hls.tcl

clean:
	rm -rf tb_preproc $(VEC) proj_preproc_accel vitis_hls.log

.PHONY: vectors test hls clean
