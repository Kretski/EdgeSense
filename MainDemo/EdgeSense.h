#ifndef EDGESENSE_H
#define EDGESENSE_H

class EdgeSense {
private:
    float kp, ki, kd;        // PID коефициенти (може да се заменят с по-сложни уравнения)
    float integral;
    float prev_error;

public:
    EdgeSense(float p = 0.8f, float i = 0.05f, float d = 0.15f)
        : kp(p), ki(i), kd(d), integral(0.0f), prev_error(0.0f) {}

    // Основна функция — EdgeSense решава raw_command
    float compute_command(float error, float sensor_value, float battery = 1.0f) {
        // Прост PID + лека зависимост от батерия и сензор
        integral += error * 0.05f;
        float derivative = error - prev_error;
        prev_error = error;

        float raw = kp * error + ki * integral + kd * derivative;

        // Пример за "ум": ако батерията е ниска — намаляваме агресивността
        if (battery < 0.3f) raw *= 0.65f;

        return raw;
    }

    void reset() {
        integral = 0.0f;
        prev_error = 0.0f;
    }
};

#endif