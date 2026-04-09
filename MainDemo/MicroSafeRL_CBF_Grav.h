#ifndef MICROSAFE_RL_CBF_GRAV_H
#define MICROSAFE_RL_CBF_GRAV_H

#include <math.h>

class MicroSafeRL_CBF_Grav {
private:
    float penalty;
    float alpha; // Параметър на бариерата

public:
    void init(float p) {
        penalty = p;
        alpha = 0.85f; // Настройка на твърдостта на бариерата
    }

    float apply_safe_control(float raw_cmd, float feedback) {
        // 🚨 КРИТИЧЕН FAIL-SAFE: Защита от счупени сензори и математически грешки (NaN)
        // Ако AI моделът или сензорът върнат Not-a-Number, спираме всичко веднага!
        if (isnan(raw_cmd) || isnan(feedback)) {
            penalty = 1.0f; // Сигнализираме за максимален системен риск
            return 0.000f;  // Пълно спиране на моторите
        }

        // --- Нормална работа на предпазителя ---
        
        // Детерминистичен CBF лимит
        float limit = 1.5f; 
        float safe_out = raw_cmd;

        // Математическа проверка на границите (Barrier Constraint)
        if (raw_cmd > limit) safe_out = limit;
        if (raw_cmd < -limit) safe_out = -limit;

        // Възстановяване на нормалния penalty статус, ако сме в безопасни граници
        penalty = 0.01f;

        // Добавяне на плавност за избягване на механичен удар
        return safe_out * alpha; 
    }

    float get_penalty() { return penalty; }
};

#endif