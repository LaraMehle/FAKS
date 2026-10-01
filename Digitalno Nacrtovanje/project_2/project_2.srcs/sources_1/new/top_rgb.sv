`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06.11.2025 08:47:08
// Design Name: 
// Module Name: top_rgb
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


module top_rgb (
    input  logic clk,       // 100 MHz clock from board
    input  logic reset,     // BTN C (center button)
    input  logic [5:0] sw,  // switches SW0-SW5
    output logic [2:0] rgb  // RGB LED outputs {B, G, R}
);
    logic [5:0] duty_r = sw[1:0] * 8;  // 0-3  -> 0,8,16,24
    logic [5:0] duty_g = sw[3:2] * 8;
    logic [5:0] duty_b = sw[5:4] * 8;

    logic pwm_r, pwm_g, pwm_b;

    pwm_rgb u_pwm (
        .clk(clk),
        .reset(reset),
        .duty_r(duty_r),
        .duty_g(duty_g),
        .duty_b(duty_b),
        .pwm_r(pwm_r),
        .pwm_g(pwm_g),
        .pwm_b(pwm_b)
    );

    // invert if LEDs are active low (na Nexys so!)
    assign rgb = ~{pwm_b, pwm_g, pwm_r};
endmodule

