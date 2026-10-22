/* EDGESENSE NANO — PLACEHOLDER MODEL
 *
 * The coefficients below come from an earlier pipeline version that fitted the
 * output of an untrained neural network instead of the logged data. They do NOT
 * describe pool_record.csv or any vessel. Kept only so MainDemo compiles.
 *
 * Regenerate with:
 *   python scripts/train_on_real_data.py --data-label SIMULATED
 * and replace this file with outputs/real_physics_model.c.
 */
#include <math.h>

float real_physics_model_inference(float x0, float x1, float x2) {
    /* x0 = error, x1 = gyro_y, x2 = thrust (unused) */
    (void)x2;
    return (x0 * (-1.0192606f) + x1 + 1.3368257f) * 0.2706757f;
}
