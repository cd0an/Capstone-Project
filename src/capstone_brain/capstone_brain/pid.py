import time


class PIDController:
    """Small reusable PID block for any node that needs time-based correction."""

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, setpoint, measured_value):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.0:
            dt = 0.01

        error = setpoint - measured_value
        p_out = self.kp * error
        self.integral += error * dt
        i_out = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative

        self.prev_error = error
        self.last_time = current_time
        return p_out + i_out + d_out

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
