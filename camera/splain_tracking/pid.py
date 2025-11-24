class PIDController:
	"""
	Prosty regulator PID do sterowania na podstawie błędu (np. error_x z obrazu).
	"""
	def __init__(self, Kp=0.01, Ki=0.0, Kd=0.0):
		self.Kp = Kp
		self.Ki = Ki
		self.Kd = Kd
		self.integral = 0.0
		self.last_error = 0.0

	def reset(self):
		self.integral = 0.0
		self.last_error = 0.0

	def compute(self, error, dt=0.1):
		"""
		:param error: aktualny błąd (np. error_x)
		:param dt: krok czasowy (domyślnie 0.1s)
		:return: sterowanie (skręt)
		"""
		self.integral += error * dt
		derivative = (error - self.last_error) / dt
		output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
		self.last_error = error
		return output
