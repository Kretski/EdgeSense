/*
 * MainDemo: EdgeSense Nano + MicroSafe-RL
 * Version: 2.1 (Final Production Build with Real Physics Model)
 */

#include "MicroSafeRL_CBF_Grav.h"
#include <math.h>

// 🧠 Декларираме НОВИЯ AI мозък (Истинската физика)
extern "C" {
    float real_physics_model_inference(float x0, float x1, float x2);
}

MicroSafeRL_CBF_Grav safety;
uint32_t start_cycle = 0;
float latency_us = 0.0f;

// ==========================================
// ⚙️ ЧЕСТНИ ХАРДУЕРНИ ПАРАМЕТРИ
// ==========================================
const float BATTERY_VOLTAGE = 12.0f; // 12V система
const float MAX_CURRENT_A = 15.0f;   // 15A мотор/ESC
const float WATT_PER_CMD = (BATTERY_VOLTAGE * MAX_CURRENT_A) / 1.5f; // ~120W на 1.0 единица команда
const float LOOP_TIME_S = 0.1f;      // 100ms (10Hz) цикъл

// Енергийна статистика
float total_energy_saved_joules = 0.0f;

void setup() {
    Serial.begin(115200);
    delay(2000); // Даваме време на серийния порт да стартира
    
    safety.init(0.01f); // Инициализация на MicroSafe-RL

    // Настройка на DWT за микросекундна отчетност
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    
    Serial.println("\n╔════════════════════════════════════════════════════════════════════════════╗");
    Serial.println("║    EdgeSense + MicroSafe-RL: REAL PHYSICS Mission Lifecycle                ║");
    Serial.println("║    Hardware Spec: 12V | 15A ESC | 180W Max Power Limit                     ║");
    Serial.println("╚════════════════════════════════════════════════════════════════════════════╝");
    Serial.println("Phase            | Raw CMD | Safe CMD | Blocked Power | Total Saved | Status");
    Serial.println("──────────────────────────────────────────────────────────────────────────────");
}

void loop() {
    start_cycle = DWT->CYCCNT;

    // Таймер на мисията (цикъл от 20 секунди)
    unsigned long current_time = millis();
    unsigned long cycle_time = current_time % 20000; 
    
    float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f;
    String phase_name = "";

    // ==========================================
    // 🌊 СТАТЕ МАШИНА: Симулация на мисията
    // ==========================================
    if (cycle_time < 4000) {
        phase_name = "Calm Water  ";
        x0 = fabs(sin(current_time / 1000.0f) * 0.5f);
        x1 = cos(current_time / 1200.0f) * 0.2f;
        x2 = 0.3f;
    } 
    else if (cycle_time < 8000) {
        phase_name = "Acceleration";
        x0 = fabs(sin(current_time / 500.0f) * 1.0f);
        x1 = cos(current_time / 500.0f) * 0.8f;
        x2 = 0.9f;
    } 
    else if (cycle_time < 14000) {
        phase_name = "Storm/Chaos ";
        x0 = fabs((sin(current_time / 200.0f) + cos(current_time / 300.0f)) * 2.5f); 
        x1 = sin(current_time / 150.0f) * 3.0f; 
        x2 = 0.6f;
    } 
    else if (cycle_time < 16000) {
        phase_name = "SENSOR FAIL ";
        x0 = NAN; // Умишлен срив (Fail-Safe тест)
        x1 = NAN;
        x2 = 0.0f;
    } 
    else {
        phase_name = "Recovery    ";
        x0 = fabs(sin(current_time / 800.0f) * 0.8f);
        x1 = cos(current_time / 900.0f) * 0.4f;
        x2 = 0.4f;
    }

    // ==========================================
    // 🧠 AI РЕШЕНИЕ И 🛡️ БЕЗОПАСНОСТ
    // ==========================================
    // 1. Извикваме ИСТИНСКАТА формула
    float raw_cmd = real_physics_model_inference(x0, x1, x2);

    // 2. Филтрираме през MicroSafe-RL
    float safe_cmd = safety.apply_safe_control(raw_cmd, x1);

    // ==========================================
    // 🔋 ЕНЕРГИЕН АНАЛИЗ (Физически модел)
    // ==========================================
    float blocked_power_watts = 0.0f;
    
    if (!isnan(raw_cmd) && (cycle_time < 14000 || cycle_time >= 16000)) {
        float expected_power = fabs(raw_cmd) * WATT_PER_CMD;
        float actual_power = fabs(safe_cmd) * WATT_PER_CMD;
        
        if (expected_power > actual_power) {
            blocked_power_watts = expected_power - actual_power;
            total_energy_saved_joules += (blocked_power_watts * LOOP_TIME_S);
        }
    } else {
        // При NaN отказ (моторът е спрян на 0.000), спестяваме номиналната мощност
        blocked_power_watts = 180.0f; 
        total_energy_saved_joules += (blocked_power_watts * LOOP_TIME_S);
    }

    // Определяне на статуса
    String status_icon;
    if (isnan(raw_cmd) || (safe_cmd == 0.000f && cycle_time >= 14000 && cycle_time < 16000)) {
        status_icon = "🚨 FAIL-SAFE";
    } else if (fabs(safe_cmd - raw_cmd) > 0.05f) {
        status_icon = "🛡️ CLIPPED";
    } else {
        status_icon = "✅ OK";
    }

    // ==========================================
    // 📊 ИЗХОД КЪМ МОНИТОРА
    // ==========================================
    Serial.print(phase_name);              Serial.print(" | ");
    
    if (raw_cmd > 99.9f) { Serial.print("MAX_ERR"); }
    else if (isnan(raw_cmd)) { Serial.print("NaN    "); }
    else { Serial.print(raw_cmd, 3); }
    Serial.print("   | ");
    
    Serial.print(safe_cmd, 3);             Serial.print("    | ");
    
    Serial.print(blocked_power_watts, 1);  Serial.print(" W     | ");
    
    if (total_energy_saved_joules > 1000.0f) {
        Serial.print(total_energy_saved_joules / 1000.0f, 2); Serial.print(" kJ   | ");
    } else {
        Serial.print(total_energy_saved_joules, 1); Serial.print(" J     | ");
    }
    
    Serial.println(status_icon);

    delay(100); // 10Hz опресняване
}