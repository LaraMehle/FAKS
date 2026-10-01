set_property SRC_FILE_INFO {cfile:{c:/Users/larci/iCloudDrive/FAKS/3. LETNIK 1. SEMESTER/DN/project_7/project_7.gen/sources_1/ip/microblaze_mcs_0/bd_0/ip/ip_0/bd_fc5c_0_microblaze_I_0.xdc} rfile:../../../project_7.gen/sources_1/ip/microblaze_mcs_0/bd_0/ip/ip_0/bd_fc5c_0_microblaze_I_0.xdc id:1 order:EARLY scoped_inst:your_instance_name/inst/microblaze_I/U0} [current_design]
set_property SRC_FILE_INFO {cfile:{C:/Users/larci/iCloudDrive/FAKS/3. LETNIK 1. SEMESTER/DN/project_7/project_7.srcs/constrs_1/imports/DN/Nexys-A7-100T-Master.xdc} rfile:../../../project_7.srcs/constrs_1/imports/DN/Nexys-A7-100T-Master.xdc id:2} [current_design]
current_instance your_instance_name/inst/microblaze_I/U0
set_property src_info {type:SCOPED_XDC file:1 line:2 export:INPUT save:INPUT read:READ} [current_design]
create_waiver -internal -scoped -user microblaze -tags 12436 -type CDC -id CDC-26 -description "Invalid LUTRAM collision warning" -to [get_pins -quiet "LOCKSTEP_Out_reg\[*\]/R"]
set_property src_info {type:SCOPED_XDC file:1 line:4 export:INPUT save:INPUT read:READ} [current_design]
create_waiver -internal -scoped -user microblaze -tags 12436 -type CDC -id CDC-26 -description "Invalid LUTRAM collision warning" -to [get_pins -quiet "MicroBlaze_Core_I/*Interrupt_DFF/Single_Synchronize.use_sync_reset.sync_reg/D"]
set_property src_info {type:SCOPED_XDC file:1 line:8 export:INPUT save:INPUT read:READ} [current_design]
create_waiver -internal -scoped -user microblaze -tags 12436 -type DRC -id PDCN-1569 -description "Valid LUT-6 instantiation" -objects [get_cells -quiet -hierarchical -filter {HLUTNM=~LUT6_2} *LUT*]
current_instance
set_property src_info {type:XDC file:2 line:8 export:INPUT save:INPUT read:READ} [current_design]
set_property -dict { PACKAGE_PIN E3    IOSTANDARD LVCMOS33 } [get_ports { clk }]; #IO_L12P_T1_MRCC_35 Sch=clk100mhz
set_property src_info {type:XDC file:2 line:75 export:INPUT save:INPUT read:READ} [current_design]
set_property -dict { PACKAGE_PIN C12   IOSTANDARD LVCMOS33 } [get_ports { reset }]; #IO_L3P_T0_DQS_AD1P_15 Sch=cpu_resetn
set_property src_info {type:XDC file:2 line:187 export:INPUT save:INPUT read:READ} [current_design]
set_property -dict { PACKAGE_PIN D4    IOSTANDARD LVCMOS33 } [get_ports { txd }]; #IO_L11N_T1_SRCC_35 Sch=uart_rxd_out
