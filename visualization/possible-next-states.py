minus = 0
plus = 0

num_houses_to_go = 3

states = {}
count = 0
num_losses = num_houses_to_go
while num_losses >= 0:
    num_increases = num_houses_to_go - num_losses
    for i in range(num_increases + 1):
        count += 1
        minus = num_losses * 10
        plus = i * 10
        state = f'h - {minus}, c + {plus}'
        print(state, f"number :{count}")

    num_losses -= 1
