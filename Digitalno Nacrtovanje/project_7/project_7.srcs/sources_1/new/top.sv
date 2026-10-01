`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04.12.2025 13:25:02
// Design Name: 
// Module Name: top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module top(
    input clk,
    input reset,
    output txd   

    );
    
    microblaze_mcs_0 your_instance_name (
      .Clk(clk),            // input wire Clk
      .Reset(reset),        // input wire Reset
      .UART_txd(txd)  // output wire UART_txd
    );

endmodule
