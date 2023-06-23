class MacAgent(object):
	def __init__(self):
		self.cur_macro_action = None
		self.cur_macro_action_done = True
		self.cur_chop_times = 0
		self.cur_macro_obs = None

	def reset(self):
		self.cur_macro_action = None
		self.cur_macro_action_done = True
		self.cur_chop_times = 0
		self.cur_macro_obs = None
	
	def get_low_level_action():
		raise

	def get_macro_action_done(self):
		raise