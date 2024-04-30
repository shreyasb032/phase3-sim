from classes.RewardModels import StateDependent

rewards_model = StateDependent()
wh = rewards_model.get_wh(health=100, time=10)
print(wh)
