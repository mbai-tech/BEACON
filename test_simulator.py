from envs.contact_simulator import simulate_contact, outcome_name

results = [simulate_contact("fragile") for _ in range(100)]
print({outcome_name(i): results.count(i) for i in range(4)})