# ============================================================================
# run_hls.tcl — Vitis HLS 2022.2 flow ของ preproc_accel  (target: Kria K26 SOM)
# ----------------------------------------------------------------------------
#   vitis_hls -f run_hls.tcl                → csim + csynth + export .xo (Vitis kernel)
#   vitis_hls -f run_hls.tcl -tclargs cosim → + RTL co-simulation (ช้า ~นาที)
#   vitis_hls -f run_hls.tcl -tclargs ip    → export เป็น IP-XACT (Vivado IPI) แทน .xo
#
# ก่อนรัน: cd hls && make vectors  (ต้องมี vectors/ ให้ csim/cosim ใช้)
# clock: 3.333 ns = 300 MHz  (DPU B4096 ใน kv260-benchmark overlay ใช้ 300/600 MHz —
#        ถ้า timing ไม่ผ่านให้ลด v++ --clock.freqHz เป็น 275 MHz ได้โดยไม่ต้องแก้ RTL)
# ============================================================================
set do_cosim  [expr {[lsearch $argv "cosim"] >= 0}]
set do_ip     [expr {[lsearch $argv "ip"] >= 0}]

open_project -reset proj_preproc_accel
set_top preproc_accel
add_files preproc_accel.cpp -cflags "-I."
add_files -tb tb_preproc.cpp -cflags "-I. -Wno-unknown-pragmas"
add_files -tb vectors

open_solution -reset "sol_k26" -flow_target [expr {$do_ip ? "vivado" : "vitis"}]
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.333 -name default
config_interface -m_axi_alignment_byte_size 64 -m_axi_max_widen_bitwidth 64
config_rtl -reset control

# csim = รัน testbench เดียวกับ g++ (ต้อง PASS ทุกเคสก่อน synth)
csim_design -argv "vectors"
csynth_design
if {$do_cosim} { cosim_design -argv "vectors 640x360" -trace_level none }
if {$do_ip} {
    export_design -format ip_catalog -output preproc_accel_ip.zip
} else {
    export_design -format xo -output preproc_accel.xo
}
exit
