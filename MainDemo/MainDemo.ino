/*
 * MainDemo: EdgeSense Nano + MicroSafe-RL
 * Version: 2.2
 *
 * Bench demo with SIMULATED inputs (no sensors, no vessel):
 *   - inputs x0 (error), x1 (gyro_y), x2 (thrust) follow a scripted mission
 *     with calm, acceleration, disturbance, sensor-failure and recovery phases;
 *   - real_physics_model_inference() produces the raw command
 *     (currently a PLACEHOLDER model, see real_symbolic_model.c);
 *   - MicroSafe-RL (MicroSafeRL.h, same file as in Kretski/MicroSafe-RL-core)
 *     filters the command;
 *   - the serial output shows raw vs filtered command, the instability penalty,
 *     and the measured latency of model + filter on the target.
 *
 * Nothing printed here is a measurement of a real vessel.
 */

#include "MicroSafeRL.h"
#include <math.h>

extern "C" {
    float real_physics_model_inference(float x0, float x1, float x2);
}

// Default MicroSafe-RL parameters, command bounds [-1.5, +1.5]
MicroSafeRL safety;

#if defined(DWT)
#define HAS_CYCLE_COUNTER 1
#else
#define HAS_CYCLE_COUNTER 0
#endif

void setup() {
    Serial.begin(115200);
    delay(2000);  // give the serial port time to come up

    safety.init(0.0f);

#if HAS_CYCLE_COUNTER
    // Enable the Cortex-M DWT cycle counter for latency measurement
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
#endif

    Serial.println();
    Serial.println("EdgeSense + MicroSafe-RL bench demo (SIMULATED inputs, placeholder model)");
    Serial.println("Phase        | Raw CMD | Safe CMD | Penalty | Latency  | Status");
    Serial.println("-------------+---------+----------+---------+----------+---------");
}

void loop() {
    // Scripted mission, 20 s cycle
    unsigned long now = millis();
    unsigned long cycle_time = now % 20000UL;

    float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f;
    const char* phase_name;

    if (cycle_time < 4000UL) {
        phase_name = "Calm        ";
        x0 = fabsf(sinf(now / 1000.0f) * 0.5f);
        x1 = cosf(now / 1200.0f) * 0.2f;
        x2 = 0.3f;
    } else if (cycle_time < 8000UL) {
        phase_name = "Acceleration";
        x0 = fabsf(sinf(now / 500.0f) * 1.0f);
        x1 = cosf(now / 500.0f) * 0.8f;
        x2 = 0.9f;
    } else if (cycle_time < 14000UL) {
        phase_name = "Disturbance ";
        x0 = fabsf((sinf(now / 200.0f) + cosf(now / 300.0f)) * 2.5f);
        x1 = sinf(now / 150.0f) * 3.0f;
        x2 = 0.6f;
    } else if (cycle_time < 16000UL) {
        phase_name = "SENSOR FAIL ";
        x0 = NAN;  // deliberate invalid input (fail-safe test)
        x1 = NAN;
        x2 = 0.0f;
    } else {
        phase_name = "Recovery    ";
        x0 = fabsf(sinf(now / 800.0f) * 0.8f);
        x1 = cosf(now / 900.0f) * 0.4f;
        x2 = 0.4f;
    }

    // --- Model + safety filter (timed section) ---
#if HAS_CYCLE_COUNTER
    uint32_t c0 = DWT->CYCCNT;
#else
    unsigned long t0 = micros();
#endif

    float raw_cmd  = real_physics_model_inference(x0, x1, x2);
    float safe_cmd = safety.apply_safe_control(raw_cmd, x1);

#if HAS_CYCLE_COUNTER
    uint32_t cycles = DWT->CYCCNT - c0;
    float latency_us = (float)cycles / ((float)SystemCoreClock / 1.0e6f);
#else
    float latency_us = (float)(micros() - t0);  // coarse: micros() resolution
#endif

    // --- Status ---
    const char* status;
    if (isnan(raw_cmd)) {
        status = "FAIL-SAFE";            // invalid command -> filter outputs 0
    } else if (fabsf(safe_cmd - raw_cmd) > 0.001f) {
        status = "LIMITED";              // filter changed the command
    } else {
        status = "PASS";
    }

    // --- Output ---
    Serial.print(phase_name);  Serial.print(" | ");
    if (isnan(raw_cmd)) { Serial.print("  NaN  "); } else { Serial.print(raw_cmd, 3); }
    Serial.print(" | ");
    Serial.print(safe_cmd, 3);               Serial.print("    | ");
    Serial.print(safety.get_penalty(), 3);   Serial.print("   | ");
    Serial.print(latency_us, 2);             Serial.print(" us  | ");
    Serial.println(status);

    delay(100);  // 10 Hz
}
