`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 23.10.2025 14:01:18
// Design Name: 
// Module Name: ALU
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


module ALU(
    input logic [5:0] a,
    input logic [5:0] b,
    input logic [3:0] alu_ctrl,
    output logic [6:0] out

    );
endmodule

    logic [5:0] add;
    logic [5:0] sub;
    logic [5:0] and_;
    logic [5:0] or_;
    logic [5:0] xor_;
    logic [5:0] equal;
    logic [5:0] funA;
    logic [5:0] funB;
    
    assign add = a + b;
    assign sub = a - b;
    assign and = a && b;
    assign or = a || b;
    assign xor = a ^ b;
    assign equal = a == b;
    
    assign out = (alu_ctrl == 1'h0) ? add :
                 (alu_ctrl == 1'h1) ? sub :
                 (alu_ctrl == 1'h2) ? and :
                 (alu_ctrl == 1'h3) ? or :
                 (alu_ctrl == 1'h4) ? xor :
                 (alu_ctrl == 1'h5) ? equal :
                 (alu_ctrl == 1'h6) ? a :
                 (alu_ctrl == 1'h7) ? b :
                 6'b000000;