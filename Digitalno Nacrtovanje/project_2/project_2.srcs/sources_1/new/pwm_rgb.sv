`timescale 1ns/1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06.11.2025 08:04:30
// Design Name: 
// Module Name: pwm_rgb
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


module pwm_rgb(
    input logic clk,
    input logic reset,
    input logic [5:0] duty_r,
    input logic [5:0] duty_g,
    input logic [5:0] duty_b,
    output logic pwm_r,
    output logic pwm_g,
    output logic pwm_b

    );
    
    logic pwm_tick;
    
    // Prescaler: šteje do 3125 (32 kHz tick)
    logic [12:0] prescaler_cnt;
    
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            prescaler_cnt <= 0;
            pwm_tick <= 0;
        end else if (prescaler_cnt == 3124) begin
            prescaler_cnt <= 0;
            pwm_tick <= 1;
        end else begin
            prescaler_cnt <= prescaler_cnt + 1;
            pwm_tick <= 0;
        end
    end
    
    // 4-bit counter - šteje do 15 in določa duty cycle
    logic [3:0] counter;
    always_ff @(posedge clk or posedge reset) begin
        if (reset)
            counter <= 4'd0;
        else if (pwm_tick)
            counter <= counter + 4'd1;
    end
    
    // Primerjava counterja in duty_cycle za generacijo PWM signala
    always_comb begin
        pwm_r = (counter < duty_r[3:0]);  // uporabi spodnje 4 bite
        pwm_g = (counter < duty_g[3:0]);
        pwm_b = (counter < duty_b[3:0]);
    end
    
endmodule
