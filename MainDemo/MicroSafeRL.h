/*
 * Copyright (c) 2026 Dimitar Kretski
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0. See the LICENSE file
 * in the repository root for the full text.
 */
#ifndef MICRO_SAFE_RL_H
#define MICRO_SAFE_RL_H

// MicroSafe-RL — lightweight deterministic output filter for AI/RL control systems.
// Reference implementation. MicroSafeRL_misra.h implements the same algorithm.
// Memory footprint: ~68 bytes (4 state floats + 8 config floats + DebugState + bool);
// verify with sizeof() on your target.
// No dynamic allocation, no loops, no recursion. C++03 compatible.

class MicroSafeRL {
public:
    // ==============================
    // DEBUG STRUCT (20 bytes)
    // ==============================
    struct DebugState {
        float instability;  // ema_mad
        float deviation;    // |sensor - mean|
        float dynamics;     // sensor velocity
        float input;        // ai_action (pre-safety)
        float output;       // final safe output
    };

    DebugState debug;

    // ==============================
    // CONSTRUCTOR
    // ==============================
    MicroSafeRL(float k = 1.15f, float a = 0.55f, float b = 2.2f,
                float lm = 0.12f, float max_p = 1.0f,
                float l_min = -1.5f, float l_max = 1.5f, float g = 0.05f)
        : ema_mean(0.0f), ema_mad(0.0f), prev_value(0.0f), current_penalty(0.0f),
          kappa(k), alpha(a), beta(b), lambda_(lm), max_penalty(max_p),
          min_limit(l_min), max_limit(l_max), gravity_factor(g),
          initialized(false)
    {
        set_debug(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }

    // ==============================
    // INIT
    // ==============================
    void init(float initial_sensor = 0.0f) {
        if (is_invalid(initial_sensor)) initial_sensor = 0.0f;
        ema_mean        = initial_sensor;
        ema_mad         = 0.0f;
        prev_value      = initial_sensor;
        current_penalty = 0.0f;
        initialized     = true;
        set_debug(0.0f, 0.0f, 0.0f, initial_sensor, initial_sensor);
    }

    // ==============================
    // CORE FUNCTION
    // ==============================
    float apply_safe_control(float ai_action, float sensor_val) {

        // --- NaN / Inf guard ---
        // Invalid action -> fail-safe zero.
        if (is_invalid(ai_action)) {
            debug.input  = ai_action;
            debug.output = 0.0f;
            return 0.0f;
        }
        // Invalid sensor -> last known good value (dead-reckoning).
        if (is_invalid(sensor_val)) {
            sensor_val = initialized ? prev_value : 0.0f;
        }

        // --- First call: auto-init, clamp only ---
        if (!initialized) {
            init(sensor_val);
            float out = clamp(ai_action);
            set_debug(0.0f, 0.0f, 0.0f, ai_action, out);
            return out;
        }

        // --- EMA mean ---
        ema_mean = lambda_ * ema_mean + (1.0f - lambda_) * sensor_val;

        // --- Deviation ---
        float abs_dev = fast_abs(sensor_val - ema_mean);

        // --- MAD (exponential) ---
        ema_mad = lambda_ * ema_mad + (1.0f - lambda_) * abs_dev;

        // --- Sensor velocity ---
        float velocity = fast_abs(sensor_val - prev_value);
        prev_value = sensor_val;

        // --- Coherence: 1 when deviation = 0, decreases with deviation ---
        float coherence = 1.0f / (1.0f + abs_dev * beta);

        // --- Instability score ---
        // 0.3 weight on velocity is a fixed design parameter (see README).
        float raw = ema_mad + alpha * (1.0f - coherence) + 0.3f * velocity;

        current_penalty = kappa * raw;
        if (current_penalty > max_penalty) current_penalty = max_penalty;
        if (current_penalty < 0.0f)        current_penalty = 0.0f;

        // --- Attenuation: G = 1 / (1 + g * P) ---
        float gain = 1.0f / (1.0f + current_penalty * gravity_factor);
        if (gain > 1.0f) gain = 1.0f;
        if (gain < 0.0f) gain = 0.0f;

        // --- Hard clamp ---
        float out = clamp(ai_action * gain);

        set_debug(ema_mad, abs_dev, velocity, ai_action, out);
        return out;
    }

    // ==============================
    // ACCESSORS
    // ==============================
    float get_penalty()        const { return current_penalty; }
    float get_current_reward() const { return 1.0f - current_penalty; }
    float get_max_penalty()    const { return max_penalty; }
    float get_min_limit()      const { return min_limit; }
    float get_max_limit()      const { return max_limit; }
    DebugState get_debug()     const { return debug; }

    // ==============================
    // RESET
    // ==============================
    void reset() {
        ema_mean        = 0.0f;
        ema_mad         = 0.0f;
        prev_value      = 0.0f;
        current_penalty = 0.0f;
        initialized     = false;
        set_debug(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }

private:
    // --- State (16 bytes) ---
    float ema_mean;
    float ema_mad;
    float prev_value;
    float current_penalty;

    // --- Configuration (32 bytes) ---
    float kappa, alpha, beta, lambda_, max_penalty;
    float min_limit, max_limit, gravity_factor;

    // --- Flags ---
    bool initialized;

    static float fast_abs(float x) { return x < 0.0f ? -x : x; }

    // NaN / Inf checks (portable, no <cmath> dependency)
    static bool is_nan(float x)     { return x != x; }
    static bool is_inf(float x)     { return x > 1e6f || x < -1e6f; }
    static bool is_invalid(float x) { return is_nan(x) || is_inf(x); }

    float clamp(float x) const {
        if (x > max_limit) return max_limit;
        if (x < min_limit) return min_limit;
        return x;
    }

    void set_debug(float inst, float dev, float dyn, float in, float out) {
        debug.instability = inst;
        debug.deviation   = dev;
        debug.dynamics    = dyn;
        debug.input       = in;
        debug.output      = out;
    }
};

#endif // MICRO_SAFE_RL_H
