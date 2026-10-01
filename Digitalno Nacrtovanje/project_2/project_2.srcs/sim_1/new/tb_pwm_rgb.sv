`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06.11.2025 08:30:10
// Design Name: 
// Module Name: tb_pwm_rgb
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

module tb_pwm_rgb;
    logic clk = 0;
    logic reset = 1;
    logic [5:0] duty_r, duty_g, duty_b;
    logic pwm_r, pwm_g, pwm_b;

    // 100 MHz clock (perioda 10 ns)
    always #5 clk = ~clk;

    // DUT (device under test)
    pwm_rgb dut (
        .clk(clk),
        .reset(reset),
        .duty_r(duty_r),
        .duty_g(duty_g),
        .duty_b(duty_b),
        .pwm_r(pwm_r),
        .pwm_g(pwm_g),
        .pwm_b(pwm_b)
    );

    initial begin
        // Reset faza
        #20 reset = 0;

        // Nastavi različne duty-cycle vrednosti
        duty_r = 6'd8;  // 50%
        duty_g = 6'd4;  // 25%
        duty_b = 6'd2;  // 12.5%

        // Počakaj malo, da vidiš PWM signale
        #200000;  // 200 µs
        $finish;
    end
endmodule
